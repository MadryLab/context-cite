# from context_citer import ContextCiter
import numpy as np
import pandas as pd
import torch as ch
import os   
from numpy.typing import NDArray
from functools import partial
from typing import Any, Optional, List, Dict, Union
# from transformers import AutoTokenizer, AutoModelForCausalLM
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
import logging
# import nltk
from spacy.lang.en import English
from tqdm.auto import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
# from transformers import DataCollatorForSeq2Seq
from groq import Groq
from openai import OpenAI
from multiprocessing import Pool
from joblib import Parallel, delayed
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import cohere

load_dotenv()

DEFAULT_GENERATE_KWARGS = {"max_new_tokens": 512, "do_sample": False}
DEFAULT_PROMPT_TEMPLATE = "Context: {context}\n\nQuery: {query}"

def _create_mask(size, alpha, seed):
    random = np.random.RandomState(seed)
    p = [1 - alpha, alpha]
    if isinstance(size, int):
        size = (size,)
    return random.choice([False, True], size=size, p=p)

def _parallel_call_groq_joblib(seed, num_sources, alpha, base_seed, context, query, partitioner, groq_model, prompt_template):
    try:
        mask = _create_mask(num_sources, alpha, seed + base_seed)
        ablated_context = partitioner.get_context(mask)
        prompt = prompt_template.format(context=ablated_context, query=query)
        messages = [{"role": "user", "content": prompt}]
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=groq_model,
        )
        response = chat_completion.choices[0].message.content
        return response
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None

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


# def _make_loader(dataset, tokenizer, batch_size):
#     # collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         collate_fn=collate_fn,
#     )
#     return loader


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


class GroqContextCiter:
    def __init__(
        self,
        groq_model: str, 
        context: str,
        query: str,
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

        if partitioner is None:
            self.partitioner = SimpleContextPartitioner(
                context, source_type=source_type
            )
        else:
            self.partitioner = partitioner
            if self.partitioner.context != context:
                raise ValueError("Partitioner context does not match provided context.")
        self.query = query
        self.generate_kwargs = generate_kwargs or DEFAULT_GENERATE_KWARGS
        self.num_ablations = num_ablations
        self.ablation_keep_prob = ablation_keep_prob
        self.batch_size = batch_size
        self.solver = solver or LassoRegression()
        self.prompt_template = prompt_template

        self._cache = {}
        self.logger = logging.getLogger("ContextCite")
        self.logger.setLevel(logging.DEBUG)  # TODO: change to INFO later

        self.groq_model = groq_model
        self.context = context

        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
        )
        
        self.cohere_client = cohere.Client(
            os.getenv("COHERE_API_KEY")
        )

        self.embedding_dim = int(os.getenv("OPENAI_EMBEDDING_DIM"))

    @property
    def response_with_indices(self, split_by="word", color=True) -> [str, pd.DataFrame]:
        """
        The response generated by the model, annotated with the starting index
        of each part.

        Arguments:
            split_by (str, optional):
                The method to split the response by. Can be "word" or "sentence".
                Defaults to "word".
            color (bool, optional):
                Whether to color the starting index of each part. Defaults to True.

        Returns:
            str:
                The response with the starting index of each part highlighted.
        """
        start_indices = []
        parts, separators, start_indices = split_text(self.response, split_by)
        separated_str = highlight_word_indices(parts, start_indices, separators, color)
        return separated_str

    @property
    def num_sources(self) -> int:
        """
        The number of sources within the context. I.e., the number of sources
        that the context is partitioned into.

        Returns:
            int:
                The number of sources in the context.
        """
        return self.partitioner.num_sources

    @property
    def sources(self) -> List[str]:
        """
        The sources within the context. I.e., the context as a list
        where each element is a source.

        Returns:
            List[str]:
                The sources within the context.
        """
        return self.partitioner.sources

    @property
    def response(self) -> str:
        return self._call_groq(context=self.context, query=self.query)

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
    
    def _get_ablated_context(self, mask):
        ablated_context = self.partitioner.get_context(mask)
        return ablated_context
    
    def _call_groq(self, context, query):
        prompt = self.prompt_template.format(context=context, query=query)
        messages = [{"role": "user", "content": prompt}] 
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
        # masks = np.zeros((num_masks, num_sources), dtype=bool)
        cosine_sims = np.zeros(self.num_ablations)
        embed_responses = np.zeros((num_masks, self.embedding_dim))
        responses = []

        masks = _create_mask(size=(num_masks, num_sources), alpha=alpha, seed=base_seed)

        # masks = ch.tensor([_create_mask(num_sources, alpha, seed + base_seed) for seed in tqdm(range(num_masks))], dtype=ch.bool)

        args = [
            (seed, num_sources, alpha, base_seed, self.context, self.query, self.partitioner, self.groq_model, self.prompt_template)
            for seed in range(num_masks)
        ]

        responses = Parallel(n_jobs=-1)(delayed(_parallel_call_groq_joblib)(*arg) for arg in tqdm(args))
        valid_indices = [i for i, response in enumerate(responses) if response is not None]
        responses = [responses[i] for i in valid_indices]
        masks = masks[valid_indices]
        parts = np.array(self.partitioner.parts)
        context_source_lists = [list(parts[mask]) for mask in masks]
        outputs = []
        for i, source in enumerate(context_source_lists):
            response = self.cohere_client.rerank(
                model="rerank-english-v3.0",
                query=self.query,
                documents=source,
                # top_n=3,
            )
            top_relevance_score = response.results[0].relevance_score
            outputs.append(top_relevance_score)

        outputs = ch.tensor(outputs, dtype=ch.float32)
        # embed_responses = self._get_embedding(responses)
        # cosine_sims = ch.nn.functional.cosine_similarity(embed_selected_response, ch.tensor(embed_responses), dim=1).numpy()
        # Save masks and cosine similarities to files
        self._visualize(masks, outputs)
        return masks, outputs

    def _visualize(self, masks, cosine_sims):
        data = {
            'context': [self.partitioner.get_context(mask) for mask in masks],
            'cosine_similarities': cosine_sims,
            'masks': masks[:, -1] #last sentence has the answer
        }
        df = pd.DataFrame(data)
        df = df.sort_values(by='cosine_similarities', ascending=False)

        # Plot scatterplot of cosine similarities with masks label to color
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df['cosine_similarities'], range(len(df)), c=df['masks'], cmap='viridis', label=df['masks'])
        plt.colorbar(scatter, label='Masks')
        plt.xlabel('Cosine Similarities')
        plt.ylabel('Index')
        plt.title('Scatterplot of Cosine Similarities with Masks Label to Color')
        plt.show()

    def _get_attributions_for_ids_range(self, start_idx, end_idx) -> tuple:
        masks, outputs = self._cosine_sim(start_idx, end_idx) # (num_ablations,)
        # num_output_tokens = end_idx - start_idx
        weight, bias = self.solver.fit_cv(masks, outputs, alphas = [0.001, 0.0001, 0.00001])
        return weight, bias
    
    def get_attributions(self, 
                        as_dataframe: bool = False,
                        top_k: Optional[int] = None,) -> NDArray:
        attributions, _bias = self._get_attributions_for_ids_range(0, len(self.response))
        if as_dataframe:
            return get_attributions_df(attributions, self.partitioner, top_k=top_k)
        else:
            return attributions

if __name__ == "__main__":
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

    cc = GroqContextCiter(groq_model='llama3-70b-8192', context=context, query=query, num_ablations=8)
    # %%
    cc.response
    # %%
    results = cc.get_attributions(as_dataframe=True, top_k=5)
    print(results.data)
    results
