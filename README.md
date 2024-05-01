## ContextCite: Attributing Model Generation to Context
<p align='center'>
    <img alt="ContextCite" src='assets/logo.png' width='75%'/>
    <br>
    [<a href="#installation">install</a>]
    [<a href="#getting-started">getting started</a>]
    [<a href="#blog-post1">blog post #1</a>]
    [<a href="#blog-post2">blog post #2</a>]
    [<a href="#demo">demo</a>]
    [<a href="#paper">paper (coming soon!)</a>]
    <!-- [<a href="#citation">Citation</a>] -->
    <br>
    Maintainers: <a href="https://claude.ai/">Aleksander Madry</a>, <a href="https://claude.ai/">Aleksander Madry</a>, and <a href="https://claude.ai/">Aleksander Madry</a>
</p>

This repository provides an API for `context-cite`, a tool for attributing LLM responses to sources within their context.

<p align = 'center'>
  <img alt="Attributing context via ContextCite" src='assets/context_cite.png' width='90%'/>
</p>


## Installation

```bash
pip install context-cite
[add requirements stuff + conda setup if needed]
```

## Getting started
Check out our [blog post](TODO), [gradio demo](TODO), and [quickstart notebook](TODO)! [add a colab notebook link too]

Using `context-cite` is as simple as:

```python
from context_cite import ContextCiter

model_name = "<huggingface model name>"
context = "..."
query = "<question relating to the context>?"

cc = ContextCiter.from_pretrained(model_name, context, query)
cc.get_attributions(as_dataframe=True)
```
Let's try the following example:

```python
from context_cite import ContextCiter

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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
cc = ContextCiter.from_pretrained(model_name, context, query)
```
We can check the model's response using
`cc.response`: `"The authors used eight P100 GPUs in their Transformer architecture for training on the WMT 2014 English-to-German translation task.</s>"`


Where did the model get its information? Let's see what the attributions look like!
```python
cc.get_attributions(as_dataframe=True, top_k=3)
```
![Example Usage](assets/readme_example.png)

