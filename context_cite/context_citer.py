import numpy as np
import pandas as pd
import torch as ch
from numpy.typing import NDArray
from typing import Dict, Any, Optional, List
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from .context_partitioner import BaseContextPartitioner, SOURCE_TYPE_TO_PARTITIONER
from .lasso import LassoRegression
from .utils import (
    get_masks_and_logit_probs,
    aggregate_logit_probs,
    split_response,
    highlight_word_indices,
    get_attributions_df,
)


DEFAULT_GENERATE_KWARGS = {"max_new_tokens": 512, "do_sample": False}
DEFAULT_PROMPT_TEMPLATE = "Context: {context}\n\nQuery: {query}"


class ContextCiter:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        context: str,
        query: str,
        source_type: str = "sentence",
        generate_kwargs: Optional[Dict[str, Any]] = None,
        num_masks: int = 64,
        ablation_keep_prob: float = 0.5,
        batch_size: int = 1,
        solver: Optional[LassoRegression] = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        partitioner_cls = SOURCE_TYPE_TO_PARTITIONER[source_type]
        self.context_partitioner: BaseContextPartitioner = partitioner_cls(context)
        self.query = query
        self.generate_kwargs = generate_kwargs or DEFAULT_GENERATE_KWARGS
        self.num_masks = num_masks
        self.ablation_keep_prob = ablation_keep_prob
        self.batch_size = batch_size
        self.solver = solver or LassoRegression()
        self.prompt_template = prompt_template

        self._cache = {}
        self.logger = logging.getLogger("ContextCite")
        self.logger.setLevel(logging.DEBUG)  # TODO: change to INFO later

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        context: str,
        query: str,
        device: str = "cuda",
        model_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
        **kwargs: Dict[str, Any],
    ) -> "ContextCiter":
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **model_kwargs
        )
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_kwargs
        )
        return cls(model, tokenizer, context, query, **kwargs)

    def _get_prompt_ids(
        self,
        mask: Optional[NDArray] = None,
        return_prompt: bool = False,
    ):
        context = self.context_partitioner.get_context(mask)
        prompt = self.prompt_template.format(context=context, query=self.query)
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        chat_prompt_ids = self.tokenizer.encode(chat_prompt, add_special_tokens=False)

        if return_prompt:
            return chat_prompt_ids, chat_prompt
        else:
            return chat_prompt_ids

    @property
    def _response_start(self):
        prompt_ids = self._get_prompt_ids()
        return len(prompt_ids)

    @property
    def _output(self):
        if self._cache.get("output") is None:
            prompt_ids, prompt = self._get_prompt_ids(return_prompt=True)
            input_ids = ch.tensor([prompt_ids], device=self.model.device)
            output_ids = self.model.generate(input_ids, **self.generate_kwargs)[0]
            # We take the original prompt because sometimes encoding and decoding changes it
            raw_output = self.tokenizer.decode(output_ids)
            prompt_length = len(self.tokenizer.decode(prompt_ids))
            self._cache["output"] = prompt + raw_output[prompt_length:]
        return self._cache["output"]

    @property
    def _output_tokens(self):
        return self.tokenizer(self._output, add_special_tokens=False)

    @property
    def _response_ids(self):
        return self._output_tokens["input_ids"][self._response_start :]

    @property
    def response(self):
        output_tokens = self._output_tokens
        char_response_start = output_tokens.token_to_chars(self._response_start).start
        return self._output[char_response_start:]

    @property
    def response_with_indices(self, split_by="word", color=True) -> [str, pd.DataFrame]:
        """
        Split the response into parts. Returns a string with the original
        response, where in front of each part the starting index of the part is
        printed in square brackets.

        Arguments:
            split_by: str, one of "word" or "sentence"
            color: bool, whether to color the starting index
        """
        start_indices = []
        parts, separators, start_indices = split_response(self.response, split_by)
        separated_str = highlight_word_indices(parts, start_indices, separators, color)
        return separated_str

    @property
    def num_sources(self) -> int:
        return self.context_partitioner.num_sources

    @property
    def sources(self) -> List[str]:
        return self.context_partitioner.sources

    def _char_range_to_token_range(self, start_index, end_index):
        output_tokens = self._output_tokens
        response_start = self._response_start
        offset = output_tokens.token_to_chars(response_start).start
        ids_start_index = output_tokens.char_to_token(start_index + offset)
        ids_end_index = output_tokens.char_to_token(end_index + offset - 1) + 1
        return ids_start_index - response_start, ids_end_index - response_start

    def _indices_to_token_indices(self, start_index=None, end_index=None):
        if start_index is None or end_index is None:
            start_index = 0
            end_index = len(self.response)
        if not (0 <= start_index < end_index <= len(self.response)):
            raise ValueError(
                f"Invalid selection range ({start_index}, {end_index}). "
                f"Please select any range within (0, {len(self.response)})."
            )
        return self._char_range_to_token_range(start_index, end_index)

    def _compute_masks_and_logit_probs(self) -> None:
        self._cache["reg_masks"], self._cache["reg_logit_probs"] = (
            get_masks_and_logit_probs(
                self.model,
                self.tokenizer,
                self.num_masks,
                self.num_sources,
                self._get_prompt_ids,
                self._response_ids,
                self.ablation_keep_prob,
                self.batch_size,
            )
        )

    @property
    def _masks(self):
        if self._cache.get("reg_masks") is None:
            self._compute_masks_and_logit_probs()
        return self._cache["reg_masks"]

    @property
    def _logit_probs(self):
        if self._cache.get("reg_logit_probs") is None:
            self._compute_masks_and_logit_probs()
        return self._cache["reg_logit_probs"]

    def _get_attributions_for_ids_range(self, ids_start_idx, ids_end_idx) -> tuple:
        outputs = aggregate_logit_probs(self._logit_probs[:, ids_start_idx:ids_end_idx])
        num_output_tokens = ids_end_idx - ids_start_idx
        weight, bias = self.solver.fit(self._masks, outputs, num_output_tokens)
        return weight, bias

    def get_attributions(
        self,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        as_dataframe: bool = False,
        top_k: Optional[int] = None,
        verbose: bool = True,
    ) -> NDArray | Any:
        """
        Get the attributions for (part of) the response.

        Arguments:
            start_idx (int, optional):
                Start index of the part to attribute to. If None, defaults to
                the start of the response.
            end_idx (int, optional):
                End index of the part to attribute to. If None, defaults to the
                end of the response.
            as_dataframe (bool, optional):
                If True, return the attributions as a stylized dataframe in
                sorted order. Otherwise, return them as a numpy array where
                the ith element corresponds to the score of the ith source
                within the context. Defaults to False.
            top_k (int, optional):
                Only used if as_dataframe is True. Number of top attributions to
                return. If None, all attributions are returned. Defaults to None.
            verbose (bool, optional):
                If True, print the selected part of the response. Defaults to
                True.

        Returns:
            NDArray | Any:
                If as_dataframe is False, return a numpy array where the ith element
                corresponds to the score of the ith source within the context.
                Otherwise, return a stylized dataframe in sorted order.
        """
        if self.num_sources == 0:
            self.logger.warning("No sources to attribute to!")
            return np.array([])

        if not as_dataframe and top_k is not None:
            self.logger.warning("top_k is ignored when not using dataframes.")

        ids_start_idx, ids_end_idx = self._indices_to_token_indices(start_idx, end_idx)
        selected_text = self.response[start_idx:end_idx]
        selected_tokens = self._response_ids[ids_start_idx:ids_end_idx]
        decoded_text = self.tokenizer.decode(selected_tokens)
        if selected_text.strip() not in decoded_text.strip():
            self.logger.warning(
                f"Decoded selected tokens do not match selected text.\n"
                f"If the following look close enough, feel free to ignore:\n"
                f"What you selected: {selected_text.strip()}\n"
                f"What is being attributed: {decoded_text.strip()}"
            )

        if verbose:
            print(f"Attributed: {decoded_text.strip()}")

        # _bias is the bias term in the l1 regression
        attributions, _bias = self._get_attributions_for_ids_range(
            ids_start_idx,
            ids_end_idx,
        )
        if as_dataframe:
            return get_attributions_df(
                attributions, self.context_partitioner, top_k=top_k
            )
        else:
            return attributions
