# Fast Causal Attributions for Rare Disease Q&A Using Groq

[Hypophosphatasia Cited Q&A App](https://context-cite-2wukcnr4uvvmpcycghrhws.streamlit.app/)

## Project Goal

The goal of this project is to leverage Groq's speed for causal attributions, a critical use case for high-stakes areas with little data such as rare diseases. Causal attribution methods such as Context-Cite enable us to identify which sentences matter for selected sentences in an answer using ablations at inference time, no training required. However, typically 32 or more ablations are required to achieve meaningful results. But by utilizing Groq's inference speed, we can achieve a >6x speedup from 30 seconds to <5 seconds at much lower cost for causal attribution generation. Additionally, we integrate a Pinecone vector database for Retrieval-Augmented Generation (RAG) on hypophosphatasia papers, and use Cohere reranking to score the importance of each sentence, making it usable for rare disease patients today.

A big limitation of this method is that Groq does not provide log probs. Context-Cite originally scores sentences based on how much they change the answer's probability. Instead, we use cosine similarity and reranking relevance score to measure how ablations affect the answer. But these methods do not work well.

For now, we have found that Cohere's reranking model by itself works better than causal attributions on Cohere's reranking relevance scores. Thus, we provide that implementation in the app to provide the best experience available.

## Next Steps
- Try other APIs w/ log probs e.g. Gemini Flash to align w/ original Context-Cite method.
- Write more general impl to support multiple APIs.

For more details on causal attributions, reference the blog post for more details from the repo this is based off of here [here](#https://gradientscience.org/contextcite/).