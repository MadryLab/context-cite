from context_citer import ContextCiter
import numpy as np
import pandas as pd
import torch as ch
import os   
from numpy.typing import NDArray
from typing import Any, Optional, List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from .context_partitioner import BaseContextPartitioner, SimpleContextPartitioner
from .solver import BaseSolver, LassoRegression
from .utils import (
    # get_masks_and_logit_probs,
    # aggregate_logit_probs,
    split_text,
    highlight_word_indices,
    get_attributions_df,
    char_to_token,
)
import nltk
from spacy.lang.en import English
from tqdm.auto import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from groq import Groq

from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)

DEFAULT_GENERATE_KWARGS = {"max_new_tokens": 512, "do_sample": False}
DEFAULT_PROMPT_TEMPLATE = "Context: {context}\n\nQuery: {query}"

def _create_mask(num_sources, alpha, seed):
    random = np.random.RandomState(seed)
    p = [1 - alpha, alpha]
    return random.choice([False, True], size=num_sources, p=p)

def get_masks_and_logit_probs(
    model,
    tokenizer,
    num_masks,
    num_sources,
    get_prompt_ids,
    response_ids,
    ablation_keep_prob,
    batch_size,
    base_seed=0,
):
    masks, dataset = _create_regression_dataset(
        num_masks,
        num_sources,
        get_prompt_ids,
        response_ids,
        ablation_keep_prob,
        base_seed=base_seed,
    )
    logit_probs = _get_response_logit_probs(
        dataset, model, tokenizer, len(response_ids), batch_size
    )
    return masks, logit_probs.astype(np.float32)

def _create_regression_dataset(
    num_masks, num_sources, get_prompt_ids, response_ids, alpha, base_seed=0
):
    masks = np.zeros((num_masks, num_sources), dtype=bool)
    data_dict = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for seed in range(num_masks):
        mask = _create_mask(num_sources, alpha, seed + base_seed)
        masks[seed] = mask
        prompt_ids = get_prompt_ids(mask=mask)
        input_ids = prompt_ids + response_ids
        data_dict["input_ids"].append(input_ids)
        data_dict["attention_mask"].append([1] * len(input_ids))
        data_dict["labels"].append([-100] * len(prompt_ids) + response_ids)
    return masks, Dataset.from_dict(data_dict)


def _compute_logit_probs(logits, labels):
    batch_size, seq_length = labels.shape
    # [num_tokens x vocab_size]
    reshaped_logits = logits.reshape(batch_size * seq_length, -1)
    reshaped_labels = labels.reshape(batch_size * seq_length)
    correct_logits = reshaped_logits.gather(-1, reshaped_labels[:, None])[:, 0]
    cloned_logits = reshaped_logits.clone()
    cloned_logits.scatter_(-1, reshaped_labels[:, None], -ch.inf)
    other_logits = cloned_logits.logsumexp(dim=-1)
    reshaped_outputs = correct_logits - other_logits
    return reshaped_outputs.reshape(batch_size, seq_length)


def _make_loader(dataset, tokenizer, batch_size):
    collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    return loader


def _get_response_logit_probs(dataset, model, tokenizer, response_length, batch_size):
    loader = _make_loader(dataset, tokenizer, batch_size)
    logit_probs = ch.zeros((len(dataset), response_length), device=model.device)

    start_index = 0
    for batch in tqdm(loader):
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with ch.no_grad(), ch.cuda.amp.autocast():
            output = model(**batch)
        logits = output.logits[:, -(response_length + 1) : -1]
        labels = batch["labels"][:, -response_length:]
        batch_size, _ = labels.shape
        cur_logit_probs = _compute_logit_probs(logits, labels)
        logit_probs[start_index : start_index + batch_size] = cur_logit_probs
        start_index += batch_size

    return logit_probs.cpu().numpy() # [num_masks x response_length]


class GroqContextCiter(ContextCiter):
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        context: str,
        query: str,
        groq, 
        source_type: str = "sentence",
        generate_kwargs: Optional[Dict[str, Any]] = None,
        num_ablations: int = 64,
        ablation_keep_prob: float = 0.5,
        batch_size: int = 1,
        solver: Optional[BaseSolver] = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        partitioner: Optional[BaseContextPartitioner] = None,
    ) -> None:
        """
        Initializes a new instance of the ContextCiter class, which is designed
        to assist in generating contextualized responses using a given machine
        learning model and tokenizer, tailored to specific queries and contexts.

        Arguments:
            model (Any):
                The model to apply ContextCite to (a HuggingFace
                ModelForCausalLM).
            tokenizer (Any):
                The tokenizer associated with the provided model.
            context (str):
                The context provided to the model
            query (str):
                The query to pose to the model.
            source_type (str, optional):
                The type of source to partition the context into. Defaults to
                "sentence", can also be "word".
            generate_kwargs (Optional[Dict[str, Any]], optional):
                Additional keyword arguments to pass to the model's generate
                method.
            num_ablations (int, optional):
                The number of ablations used to train the surrogate model.
                Defaults to 64.
            ablation_keep_prob (float, optional):
                The probability of keeping a source when ablating the context.
                Defaults to 0.5.
            batch_size (int, optional):
                The batch size used when performing inference using ablated
                contexts. Defaults to 1.
            solver (Optional[Solver], optional):
                The solver to use to compute the linear surrogate model. Lasso
                regression is used by default.
            prompt_template (str, optional):
                A template string used to create the prompt from the context
                and query.
            partitioner (Optional[BaseContextPartitioner], optional):
                A custom partitioner to split the context into sources. This
                will override "source_type" if specified.
        """
        super().__init__()
        self._groq = []


    def _compute_masks_and_logit_probs(self) -> None:
        self._cache["reg_masks"], self._cache["reg_logit_probs"] = (
            get_masks_and_logit_probs(
                self.model,
                self.tokenizer,
                self.num_ablations,
                self.num_sources,
                self._get_prompt_ids,
                self._response_ids,
                self.ablation_keep_prob,
                self.batch_size,
            )
        )