import numpy as np
import pandas as pd
import torch as ch
from numpy.typing import NDArray
from typing import Any, Optional, List, Dict, Union
from context_cite.context_partitioner import BaseContextPartitioner, SimpleContextPartitioner
from context_cite.solver import BaseSolver, LassoRegression, CosineSimLassoRegression
from context_cite.utils import (
    split_text,
    highlight_word_indices,
    get_attributions_df,
)
import logging
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

DEFAULT_GENERATE_KWARGS = {"max_new_tokens": 512, "do_sample": False}
DEFAULT_PROMPT_TEMPLATE = "Context: {context}\n\nQuery: {query}"

def _create_mask(size, alpha, seed):
    random = np.random.RandomState(seed)
    p = [1 - alpha, alpha]
    if isinstance(size, int):
        size = (size,)
    return random.choice([False, True], size=size, p=p)

def _parallel_call_groq_joblib(seed, num_sources, alpha, base_seed, query, partitioner, groq_model, groq_client, prompt_template):
    try:
        mask = _create_mask(num_sources, alpha, seed + base_seed)
        ablated_context = partitioner.get_context(mask)
        prompt = prompt_template.format(context=ablated_context, query=query)
        messages = [{"role": "user", "content": prompt}]
        
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=groq_model,
        )
        response = chat_completion.choices[0].message.content
        return response
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None
class GroqContextCiter:
    def __init__(
        self,
        groq_model: str, 
        context: str,
        query: str,
        groq_client,
        cohere_client,
        openai_client,
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

        self.context = context
        self.groq_model = groq_model
        self.groq_client = groq_client
        self.cohere_client = cohere_client
        self.openai_client = openai_client
        self.embedding_dim = 256 #hardcoded to match our vector db for hpp

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
    def causal_reranking(self, selected_response):
        # selected_response = self.response[start_idx:end_idx]
        num_masks = self.num_ablations
        num_sources = self.num_sources
        alpha = self.ablation_keep_prob
        base_seed = 0
        responses = []

        masks = _create_mask(size=(num_masks, num_sources), alpha=alpha, seed=base_seed)

        # masks = ch.tensor([_create_mask(num_sources, alpha, seed + base_seed) for seed in tqdm(range(num_masks))], dtype=ch.bool)

        args = [
            (seed, num_sources, alpha, base_seed,  self.query, self.partitioner, self.groq_model, self.groq_client, self.prompt_template)
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
            if not source:
                outputs.append(0)
                continue
            response = self.cohere_client.rerank(
                model="rerank-english-v3.0",
                query=self.query,
                documents=source,
                top_n=1,
            )
            top_relevance_score = response.results[0].relevance_score
            outputs.append(top_relevance_score)

        outputs = ch.tensor(outputs, dtype=ch.float32)
        # embed_responses = self._get_embedding(responses)
        # cosine_sims = ch.nn.functional.cosine_similarity(embed_selected_response, ch.tensor(embed_responses), dim=1).numpy()
        # Save masks and cosine similarities to files
        # self._visualize(masks, outputs)
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

    def _get_attributions_for_ids_range(self, sentence) -> tuple:
        masks, outputs = self.causal_reranking(sentence) # (num_ablations,)
        # num_output_tokens = end_idx - start_idx
        weight, bias = self.solver.fit_cv(masks, outputs, alphas = [0.001, 0.0001, 0.00001])
        return weight, bias
    
    def get_attributions(self, sentence,
                        as_dataframe: bool = False,
                        top_k: Optional[int] = None,) -> NDArray:
        attributions, _bias = self._get_attributions_for_ids_range(sentence)
        if as_dataframe:
            return get_attributions_df(attributions, self.partitioner, top_k=top_k)
        else:
            return attributions
