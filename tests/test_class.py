import pytest
import numpy as np
from typing import Any
from context_cite import ContextCiter
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAMES = [
    # "mistralai/Mistral-7B-Instruct-v0.2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


def get_model(model_name: str) -> tuple[Any, Any]:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_context_and_query() -> tuple[str, str]:
    context = "The quick brown fox named Charlie jumps over the lazy dog named John."
    query = "What is the name of the quick brown fox?"
    return context, query


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_class_init(model_name) -> None:
    context, query = get_context_and_query()

    model, tokenizer = get_model(model_name)
    cc = ContextCiter(model, tokenizer, context, query)
    assert isinstance(cc, ContextCiter)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_class_init_from_pretrained(model_name) -> None:
    context, query = get_context_and_query()

    cc = ContextCiter.from_pretrained(model_name, context, query)
    assert isinstance(cc, ContextCiter)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_generate(model_name) -> None:
    model, tokenizer = get_model(model_name)
    model.eval()
    model.cuda()
    context, query = get_context_and_query()
    cc = ContextCiter(model, tokenizer, context, query)

    R = cc.response
    print(R)
    assert isinstance(R, str)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_attribute_all(model_name) -> None:
    model, tokenizer = get_model(model_name)
    model.eval()
    model.cuda()
    context, query = get_context_and_query()
    cc = ContextCiter(model, tokenizer, context, query, num_masks=16)
    scores = cc.get_attributions()
    assert isinstance(scores, np.ndarray)
