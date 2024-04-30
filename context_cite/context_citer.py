import numpy as np
import pandas as pd
import torch as ch
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from .context_partitioner import BaseContextPartitioner, PARTITION_TYPE_TO_PARTITIONER
from .lasso import LassoRegression
from .utils import (
    get_masks_and_logit_probs,
    aggregate_logit_probs,
    split_into_sentences,
    split_into_words,
    highlight_word_indices,
    get_formatted_scores_and_sources,
)


DEFAULT_GENERATE_KWARGS = {"max_new_tokens": 512, "do_sample": False}


class ContextCiter:
    def __init__(
        self,
        model,
        tokenizer,
        context: str,
        query: str,
        partition_type: str = "sentence",
        generate_kwargs: Optional[Dict[str, Any]] = None,
        num_masks: int = 64,
        ablation_keep_prob: float = 0.5,
        batch_size: int = 1,
        solver: LassoRegression = LassoRegression(lasso_alpha=0.01),
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        partitioner_cls = PARTITION_TYPE_TO_PARTITIONER[partition_type]
        self.context_partitioner: BaseContextPartitioner = partitioner_cls(context)
        self.query = query
        self.generate_kwargs = generate_kwargs or DEFAULT_GENERATE_KWARGS
        self.num_masks = num_masks
        self.ablation_keep_prob = ablation_keep_prob
        self.batch_size = batch_size
        self.solver = solver
        self._cache = {}

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        context: str,
        query: str,
        partition_type: str = "sentence",
        device: str = "cuda",
        model_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
        generate_kwargs: Optional[Dict[str, Any]] = None,
    ):
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **model_kwargs
        )
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_kwargs
        )
        return cls(model, tokenizer, context, query, partition_type, generate_kwargs)

    def _get_prompt_and_ids(self, mask: Optional[np.ndarray] = None):
        context = self.context_partitioner.get_context(mask)
        prompt = f"Context: {context}\n\nQuery: {self.query}"
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        chat_prompt_ids = self.tokenizer.encode(chat_prompt, add_special_tokens=False)
        return chat_prompt, chat_prompt_ids

    @property
    def _response_start(self):
        _, prompt_ids = self._get_prompt_and_ids()
        return len(prompt_ids)

    @property
    def _output(self):
        if self._cache.get("output") is None:
            prompt, prompt_ids = self._get_prompt_and_ids()
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
        # 32007 is a special token for Phi-3 denoting <|end|>
        # 128009 is a special token for Llama3 denoting <|eot_id|>
        if output_tokens["input_ids"][-1] in (
            self.tokenizer.eos_token_id,
            32007,
            128009,
        ):
            eos_start = output_tokens.token_to_chars(
                len(output_tokens["input_ids"]) - 1
            ).start
            return self._output[char_response_start:eos_start]
        else:
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
        if split_by == "word":
            parts, separators = split_into_words(self.response)
        elif split_by == "sentence":
            parts, separators = split_into_sentences(self.response)
        else:
            raise ValueError(f"Invalid split_by: {split_by}")

        cur_start_index = 0
        for separator, part in zip(separators, parts):
            start_indices.append(cur_start_index)
            cur_start_index += len(separator) + len(part)

        separated_str = highlight_word_indices(parts, start_indices, separators, color)
        return separated_str

    @property
    def num_sources(self):
        return self.context_partitioner.num_sources

    @property
    def sources(self):
        return self.context_partitioner.sources

    def get_source(self, index):
        return self.context_partitioner.get_source(index)

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
        assert 0 <= start_index
        assert start_index < end_index
        assert end_index <= len(self.response)
        ids_start_index, ids_end_index = self._char_range_to_token_range(
            start_index, end_index
        )
        return ids_start_index, ids_end_index

    def _get_masks_and_logit_probs(self):
        def get_prompt_ids(mask=None):
            _, prompt_ids = self._get_prompt_and_ids(mask=mask)
            return prompt_ids

        return get_masks_and_logit_probs(
            self.model,
            self.tokenizer,
            self.num_masks,
            self.num_sources,
            get_prompt_ids,
            self._response_ids,
            self.ablation_keep_prob,
            self.batch_size,
        )

    @property
    def _masks(self):
        if self._cache.get("reg_masks") is None:
            self._cache["reg_masks"], self._cache["reg_logit_probs"] = (
                self._get_masks_and_logit_probs()
            )
        return self._cache["reg_masks"]

    @property
    def _logit_probs(self):
        if self._cache.get("reg_logit_probs") is None:
            self._cache["reg_masks"], self._cache["reg_logit_probs"] = (
                self._get_masks_and_logit_probs()
            )
        return self._cache["reg_logit_probs"]

    def _get_scores_for_ids_range(
        self, ids_start_idx, ids_end_idx, return_extras=False
    ):

        outputs = aggregate_logit_probs(self._logit_probs[:, ids_start_idx:ids_end_idx])
        num_output_tokens = ids_end_idx - ids_start_idx
        weight, bias = self.solver.fit(self._masks, outputs, num_output_tokens)
        extras = {"bias": bias}
        if return_extras:
            return weight, extras
        else:
            return weight

    def get_attributions(self, start_index=None, end_index=None, return_extras=False):
        if self.num_sources == 0:
            return np.array([])
        ids_start_idx, ids_end_idx = self._indices_to_token_indices(
            start_index, end_index
        )
        selected = self.response[start_index:end_index]
        attributed = self.tokenizer.decode(
            self._response_ids[ids_start_idx:ids_end_idx]
        )
        assert selected.strip() in attributed.strip()
        return self._get_scores_for_ids_range(
            ids_start_idx,
            ids_end_idx,
            return_extras=return_extras,
        )

    def show_attributions(self, start_idx=None, end_idx=None):
        scores = self.get_attributions(start_index=start_idx, end_index=end_idx)
        order = scores.argsort()[::-1]
        selected_scores = []
        selected_sources = []
        for i in order:
            selected_scores.append(scores[i])
            selected_sources.append(self.get_source(i))
        df = get_formatted_scores_and_sources(selected_scores, selected_sources)
        return df
