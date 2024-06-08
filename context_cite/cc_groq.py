from context_citer import ContextCiter
import numpy as np
import pandas as pd
import torch as ch
import os   
from numpy.typing import NDArray
from typing import Any, Optional, List, Dict, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from context_cite.context_partitioner import BaseContextPartitioner, SimpleContextPartitioner
from context_cite.solver import BaseSolver, LassoRegression, CosineSimLassoRegression
from context_cite.utils import (
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
from openai import OpenAI


from dotenv import load_dotenv

load_dotenv()

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

    data_dict = {
        "sentences": [],
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
        groq_model: str = 'llama3-8b-8192', 
        source_type: str = "sentence",
        generate_kwargs: Optional[Dict[str, Any]] = None,
        num_ablations: int = 64,
        ablation_keep_prob: float = 0.5,
        batch_size: int = 1,
        solver: Optional[BaseSolver] = CosineSimLassoRegression(),
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
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            context=context,
            query=query,
            source_type=source_type,
            generate_kwargs=generate_kwargs,
            num_ablations=num_ablations,
            ablation_keep_prob=ablation_keep_prob,
            batch_size=batch_size,
            solver=solver,
            prompt_template=prompt_template,
            partitioner=partitioner,
        )
        self.groq_model = groq_model

        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
        )

        self.embedding_dim = int(os.getenv("OPENAI_EMBEDDING_DIM"))

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

    def _get_embedding(self, text: Union[str, List[str]]) -> ch.Tensor:
        if isinstance(text, str):
            text = [text]
        embedding_response = self.openai_client.embeddings.create(input=text, 
                                                         model="text-embedding-3-small", 
                                                         dimensions=self.embedding_dim)
        embeddings = ch.stack([ch.tensor(item.embedding) for item in embedding_response.data])
        if embeddings.dim() == 1:
            return embeddings.unsqueeze(0)
        return embeddings
    def _get_prompt_ids_groq(self, mask):
        context = self.partitioner.get_context(mask)
        prompt = self.prompt_template.format(context=context, query=self.query)
        messages = [{"role": "user", "content": prompt}]
        return messages
    
    def _call_groq(self, messages):
        chat_completion = self.groq_client.chat.completions.create(
            messages=messages,
            model=self.groq_model,
        )
        return chat_completion.choices[0].message.content

    # @property
    def _cosine_sim(self, start_idx, end_idx):
        selected_response = self.response[start_idx:end_idx]
        embed_selected_response = self._get_embedding(selected_response)

        num_masks = self.num_ablations
        num_sources = self.num_sources
        alpha = self.ablation_keep_prob
        base_seed = 0
        masks = np.zeros((num_masks, num_sources), dtype=bool)
        cosine_sims = np.zeros(self.num_ablations)
        embed_responses = np.zeros((num_masks, self.embedding_dim))
        responses = []

        for seed in tqdm(range(num_masks)):
            mask = _create_mask(num_sources, alpha, seed + base_seed)
            masks[seed] = mask
            messages = self._get_prompt_ids_groq(mask=mask)
            response = self._call_groq(messages)
            responses.append(response)

        embed_responses = self._get_embedding(responses)        
        cosine_sims = ch.nn.functional.cosine_similarity(embed_selected_response, ch.tensor(embed_responses), dim=1).numpy()
        return masks, cosine_sims

    def _get_attributions_for_ids_range(self, start_idx, end_idx) -> tuple:
        masks, outputs = self._cosine_sim(start_idx, end_idx) # (num_ablations,)
        # num_output_tokens = end_idx - start_idx
        weight, bias = self.solver.fit(masks, outputs)
        return weight, bias
    
    def get_attributions(self, 
                        as_dataframe: bool = False,
                        top_k: Optional[int] = None,) -> NDArray:
        attributions, _bias = self._get_attributions_for_ids_range(0, len(self.response))
        if as_dataframe:
            return get_attributions_df(attributions, self.partitioner, top_k=top_k)
        else:
            return attributions

model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

context = """
Attention Is All You Need

Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
1 Introduction
Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].
Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht-1 and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.
Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.
In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.
"""
query = "What type of GPUs did the authors use in this paper?"

cc = GroqContextCiter.from_pretrained(model_name_or_path, context, query, device='cpu')
# %%
cc.response
# %%
results = cc.get_attributions(as_dataframe=True, top_k=5)
print(results.data)
results