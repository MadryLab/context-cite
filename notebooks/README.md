## Example notebooks

Try out `context_cite` using our example notebooks (you can open them in Google colab).

### [🤗 Quickstart](https://github.com/MadryLab/context-cite/blob/main/notebooks/quickstart_example.ipynb) <a target="_blank" href="https://colab.research.google.com/github/MadryLab/context-cite/blob/main/notebooks/quickstart_example.ipynb"><img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>

In this notebook, we walk through the core functionality of `context_cite` by applying it to a TinyLlama chat model to answer a question about the paper "Attention Is All You Need."
We create a `ContextCiter` and provide it with (1) the model and tokenizer (2) the abstract and introduction of the paper and (3) the query `"What type of GPUs did the authors use in this paper?"`. 
From there, applying `context_cite` is as easy as viewing the `response` property to see the model's response and calling `get_attributions` to display the sources it used for either the entire response or any selection from the response.

### [🦜⛓️ RAG example](https://github.com/MadryLab/context-cite/blob/main/notebooks/rag_langchain_example.ipynb) <a target="_blank" href="https://colab.research.google.com/github/MadryLab/context-cite/blob/main/notebooks/rag_langchain_example.ipynb"> <img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

In practice, there is often a retrieval step to select the information to provide to an LLM in its context (this process is called retrieval-augmented generation or RAG).
In this notebook, we use `langchain` to apply `context_cite` to a in-context information selected with RAG.
We first create a simple RAG chain without `context_cite` and then create a `context_cite` chain to get citations for the response.