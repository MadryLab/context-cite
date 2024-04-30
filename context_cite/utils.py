import nltk
import numpy as np
import pandas as pd
import torch as ch
from tqdm.auto import tqdm
from typing import List, Tuple
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

nltk.download("punkt")


def split_into_sentences(text: str) -> Tuple[List[str], List[str]]:
    """Split text into sentences and return the sentences and separators."""
    sentences = []
    separators = []

    # first split by newlines
    lines = text.splitlines()
    for line in lines:
        sentences.extend(nltk.sent_tokenize(line))

    cur_start = 0
    for sentence in sentences:
        cur_end = text.find(sentence, cur_start)
        separators.append(text[cur_start:cur_end])
        cur_start = cur_end + len(sentence)
    return sentences, separators


def split_into_words(text: str) -> Tuple[List[str], List[str]]:
    """Split text into words and return the words and separators."""
    words = nltk.word_tokenize(text)
    separators = []
    cur_start = 0
    for word in words:
        cur_end = text.find(word, cur_start)
        separators.append(text[cur_start:cur_end])
        cur_start = cur_end + len(word)
    return words, separators


def highlight_word_indices(words, indices, separators, color: bool):
    formatted_words = []

    # ANSI escape code for red color
    if color:
        # RED = "\033[91m"
        RED = "\033[36m"  # ANSI escape code for light gray
        RESET = "\033[0m"  # Reset color to default
    else:
        RED = ""
        RESET = ""

    for word, idx in zip(words, indices):
        # Wrap index with red color
        formatted_words.append(f"{RED}[{idx}]{RESET}{word}")

    result = "".join(sep + word for sep, word in zip(separators, formatted_words))
    return result


def create_mask(num_sources, alpha, seed):
    random = np.random.RandomState(seed)
    p = [1 - alpha, alpha]
    return random.choice([False, True], size=num_sources, p=p)


def create_regression_dataset(
    num_masks, num_sources, get_prompt_ids, response_ids, alpha, base_seed=0
):
    masks = np.zeros((num_masks, num_sources), dtype=bool)
    data_dict = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for seed in range(num_masks):
        mask = create_mask(num_sources, alpha, seed + base_seed)
        masks[seed] = mask
        prompt_ids = get_prompt_ids(mask=mask)
        input_ids = prompt_ids + response_ids
        data_dict["input_ids"].append(input_ids)
        data_dict["attention_mask"].append([1] * len(input_ids))
        data_dict["labels"].append([-100] * len(prompt_ids) + response_ids)
    return masks, Dataset.from_dict(data_dict)


def compute_logit_probs(logits, labels):
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


def make_loader(dataset, tokenizer, batch_size):
    collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    return loader


def get_response_logit_probs(dataset, model, tokenizer, response_length, batch_size):
    loader = make_loader(dataset, tokenizer, batch_size)
    logit_probs = ch.zeros((len(dataset), response_length), device=model.device)

    start_index = 0
    for batch in tqdm(loader):
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with ch.no_grad(), ch.cuda.amp.autocast():
            output = model(**batch)
        logits = output.logits[:, -(response_length + 1) : -1]
        labels = batch["labels"][:, -response_length:]
        batch_size, _ = labels.shape
        cur_logit_probs = compute_logit_probs(logits, labels)
        logit_probs[start_index : start_index + batch_size] = cur_logit_probs
        start_index += batch_size

    return logit_probs.cpu().numpy()


def get_masks_and_logit_probs(
    model,
    tokenizer,
    num_masks,
    num_sources,
    get_prompt_ids,
    response_ids,
    alpha,
    batch_size,
    base_seed=0,
):
    masks, dataset = create_regression_dataset(
        num_masks, num_sources, get_prompt_ids, response_ids, alpha, base_seed=base_seed
    )
    logit_probs = get_response_logit_probs(
        dataset, model, tokenizer, len(response_ids), batch_size
    )
    return masks, logit_probs.astype(np.float32)


def aggregate_logit_probs(logit_probs, output_type="logit_prob"):
    """Compute sequence-level outputs from token-level logit-probabilities."""
    logit_probs = ch.tensor(logit_probs)
    log_probs = ch.nn.functional.logsigmoid(logit_probs).sum(dim=1)
    if output_type == "log_prob":
        return log_probs.numpy()
    elif output_type == "logit_prob":
        log_1mprobs = ch.log1p(-ch.exp(log_probs))
        return (log_probs - log_1mprobs).numpy()
    elif output_type == "total_token_logit_prob":
        return logit_probs.mean(dim=1).numpy()
    else:
        raise ValueError(f"Cannot aggregate log probs for output type '{output_type}'")


def _color_scale(val, max_val):
    start_color = (255, 255, 255)
    end_color = (80, 180, 80)
    if val == 0:
        return f"background-color: rgb{start_color}"
    elif val == max_val:
        return f"background-color: rgb{end_color}"
    else:
        fraction = val / max_val
        interpolated_color = tuple(
            start_color[i] + (end_color[i] - start_color[i]) * fraction
            for i in range(3)
        )
        return f"background-color: rgb{interpolated_color}"


def _apply_color_scale(df):
    max_val = max([df["Score"].max(), 1])
    return df.style.map(lambda val: _color_scale(val, max_val), subset=["Score"])


def get_formatted_scores_and_sources(scores, sources):
    df = pd.DataFrame.from_dict({"Score": scores, "Source": sources})
    return _apply_color_scale(df).format(precision=3)
