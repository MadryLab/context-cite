import pytest
import numpy as np
import torch as ch
import pandas as pd
from typing import Any
from context_cite import ContextCiter
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAMES = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/Phi-3-mini-128k-instruct",
]


def get_model(model_name: str) -> tuple[Any, Any]:
    # trust_remote_code=True is necessary for the Phi-3 model
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def get_context_and_query(long=False) -> tuple[str, str]:
    if long:
        context = """
                Attention Is All You Need

                Abstract
                The dominant sequence transduction models are based on complex recurrent or
                convolutional neural networks that include an encoder and a decoder. The best
                performing models also connect the encoder and decoder through an attention
                mechanism. We propose a new simple network architecture, the Transformer, based
                solely on attention mechanisms, dispensing with recurrence and convolutions
                entirely. Experiments on two machine translation tasks show these models to be
                superior in quality while being more parallelizable and requiring significantly
                less time to train. Our model achieves 28.4 BLEU on the WMT 2014
                English-to-German translation task, improving over the existing best results,
                including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French
                translation task, our model establishes a new single-model state-of-the-art BLEU
                score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the
                training costs of the best models from the literature. We show that the
                Transformer generalizes well to other tasks by applying it successfully to
                English constituency parsing both with large and limited training data.

                1 Introduction
                Recurrent neural networks, long short-term memory [13] and gated
                recurrent [7] neural networks in particular, have been firmly
                established as state of the art approaches in sequence modeling and
                transduction problems such as language modeling and machine translation
                [35, 2, 5]. Numerous efforts have since continued to push the boundaries
                of recurrent language models and encoder-decoder architectures [38, 24,
                15].  Recurrent models typically factor computation along the symbol
                positions of the input and output sequences. Aligning the positions to
                steps in computation time, they generate a sequence of hidden states ht,
                as a function of the previous hidden state ht-1 and the input for
                position t. This inherently sequential nature precludes parallelization
                within training examples, which becomes critical at longer sequence
                lengths, as memory constraints limit batching across examples. Recent
                work has achieved significant improvements in computational efficiency
                through factorization tricks [21] and conditional computation [32],
                while also improving model performance in case of the latter. The
                fundamental constraint of sequential computation, however, remains.
                Attention mechanisms have become an integral part of compelling sequence
                modeling and transduction models in various tasks, allowing modeling of
                dependencies without regard to their distance in the input or output
                sequences [2, 19]. In all but a few cases [27], however, such attention
                mechanisms are used in conjunction with a recurrent network.  In this
                work we propose the Transformer, a model architecture eschewing
                recurrence and instead relying entirely on an attention mechanism to
                draw global dependencies between input and output. The Transformer
                allows for significantly more parallelization and can reach a new state
                of the art in translation quality after being trained for as little as
                twelve hours on eight P100 GPUs.
        """
        query = "What type of GPUs did the authors use in this paper?"

    else:
        context = (
            "The quick brown fox named Charlie jumps over the lazy dog named John."
        )
        query = "What is the name of the quick brown fox?"

    return context, query


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_class_init(model_name: str) -> None:
    context, query = get_context_and_query()

    model, tokenizer = get_model(model_name)
    cc = ContextCiter(model, tokenizer, context, query)
    assert isinstance(cc, ContextCiter)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_class_init_from_pretrained(model_name: str) -> None:
    context, query = get_context_and_query()

    cc = ContextCiter.from_pretrained(
        model_name,
        context,
        query,
        model_kwargs={"trust_remote_code": True},
        tokenizer_kwargs={"trust_remote_code": True},
    )
    assert isinstance(cc, ContextCiter)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_class_init_from_pretrained_cpu(model_name: str) -> None:
    context, query = get_context_and_query()

    cc = ContextCiter.from_pretrained(
        model_name,
        context,
        query,
        device="cpu",
        model_kwargs={"trust_remote_code": True},
        tokenizer_kwargs={"trust_remote_code": True},
    )
    assert isinstance(cc, ContextCiter)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_generate(model_name: str) -> None:
    model, tokenizer = get_model(model_name)
    model.eval()
    model.cuda()
    context, query = get_context_and_query()
    cc = ContextCiter(model, tokenizer, context, query)

    R = cc.response
    print(R)
    assert isinstance(R, str)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_generate_long(model_name: str) -> None:
    model, tokenizer = get_model(model_name)
    model.eval()
    model.cuda()
    context, query = get_context_and_query(long=True)
    cc = ContextCiter(model, tokenizer, context, query)

    R = cc.response
    print(R)
    assert isinstance(R, str)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_generate_on_cpu(model_name: str) -> None:
    model, tokenizer = get_model(model_name)
    model.eval()
    context, query = get_context_and_query()
    cc = ContextCiter(model, tokenizer, context, query)

    R = cc.response
    print(R)
    assert isinstance(R, str)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_generate_on_cpu_from_pretrained(model_name: str) -> None:
    context, query = get_context_and_query()
    cc = ContextCiter.from_pretrained(
        model_name,
        context,
        query,
        device="cpu",
        model_kwargs={"trust_remote_code": True},
        tokenizer_kwargs={"trust_remote_code": True},
    )

    R = cc.response
    print(R)
    assert isinstance(R, str)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_generate_same_without_contextcite(model_name: str) -> None:
    from context_cite.context_citer import (
        DEFAULT_PROMPT_TEMPLATE,
        DEFAULT_GENERATE_KWARGS,
    )

    context, query = get_context_and_query()

    cc = ContextCiter.from_pretrained(
        model_name,
        context,
        query,
        device="cpu",
        model_kwargs={"trust_remote_code": True},
        tokenizer_kwargs={"trust_remote_code": True},
    )

    contextcite_response = cc.response

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    prompt = DEFAULT_PROMPT_TEMPLATE.format(context=context, query=query)
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    chat_prompt_ids = tokenizer.encode(chat_prompt, add_special_tokens=False)
    input_ids = ch.tensor([chat_prompt_ids], device=model.device)
    output_ids = model.generate(input_ids, **DEFAULT_GENERATE_KWARGS)[0]
    raw_output = tokenizer.decode(output_ids)
    prompt_length = len(tokenizer.decode(chat_prompt_ids))
    output = chat_prompt + raw_output[prompt_length:]
    output_tokens = tokenizer(output, add_special_tokens=False)
    response_start = len(chat_prompt_ids)
    char_response_start = output_tokens.token_to_chars(response_start).start
    assert contextcite_response == output[char_response_start:]


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_attribute_all(model_name: str) -> None:
    model, tokenizer = get_model(model_name)
    model.eval()
    model.cuda()
    context, query = get_context_and_query()
    cc = ContextCiter(model, tokenizer, context, query, num_ablations=16)
    scores = cc.get_attributions()
    assert isinstance(scores, np.ndarray)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_attribute_pretty(model_name: str) -> None:
    model, tokenizer = get_model(model_name)
    model.eval()
    model.cuda()
    context, query = get_context_and_query()
    cc = ContextCiter(model, tokenizer, context, query, num_ablations=16)
    scores = cc.get_attributions(as_dataframe=True)
    assert isinstance(scores.data, pd.DataFrame)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_attribute_pretty_topk(model_name: str) -> None:
    model, tokenizer = get_model(model_name)
    model.eval()
    model.cuda()
    context, query = get_context_and_query(long=True)
    cc = ContextCiter(model, tokenizer, context, query, num_ablations=16)
    scores = cc.get_attributions(as_dataframe=True, top_k=5)
    assert isinstance(scores.data, pd.DataFrame)
    assert len(scores.data.index) == min(5, cc.num_sources)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_attribute_long(model_name: str) -> None:
    model, tokenizer = get_model(model_name)
    model.eval()
    model.cuda()
    context, query = get_context_and_query(long=True)
    cc = ContextCiter(model, tokenizer, context, query, num_ablations=16)
    scores = cc.get_attributions()
    assert isinstance(scores, np.ndarray)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_attribute_partial(model_name: str) -> None:
    model, tokenizer = get_model(model_name)
    model.eval()
    model.cuda()
    context, query = get_context_and_query()
    cc = ContextCiter(model, tokenizer, context, query, num_ablations=16)
    R = cc.response

    start_idx = np.random.randint(0, len(R) - 1)
    end_idx = np.random.randint(start_idx + 1, len(R))
    scores = cc.get_attributions(start_idx, end_idx)

    assert isinstance(scores, np.ndarray)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_attribute_word(model_name: str) -> None:
    model, tokenizer = get_model(model_name)
    model.eval()
    model.cuda()
    context, query = get_context_and_query()
    cc = ContextCiter(
        model, tokenizer, context, query, num_ablations=16, source_type="word"
    )
    R = cc.response

    start_idx = np.random.randint(0, len(R) - 1)
    end_idx = np.random.randint(start_idx + 1, len(R))
    scores = cc.get_attributions(start_idx, end_idx)

    assert isinstance(scores, np.ndarray)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_attribute_word_from_pretrained(model_name: str) -> None:
    context, query = get_context_and_query()
    cc = ContextCiter.from_pretrained(
        model_name,
        context,
        query,
        source_type="word",
        model_kwargs={"trust_remote_code": True},
        tokenizer_kwargs={"trust_remote_code": True},
    )
    R = cc.response

    start_idx = np.random.randint(0, len(R) - 1)
    end_idx = np.random.randint(start_idx + 1, len(R))
    scores = cc.get_attributions(start_idx, end_idx)

    assert isinstance(scores, np.ndarray)
