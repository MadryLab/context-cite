Fast Causal Attributions for Rare Disease Q&A Using Groq

## Project Goal

The goal of this project is to leverage Groq's speed for causal attributions, a critical use case for high-stakes areas with little data such as rare diseases. Causal attribution methods such as Context-Cite enable us to identify which sentences matter for selected sentences in an answer using ablations at inference time, no training required. However, typically 32 or more ablations are required to achieve meaningful results. But by utilizing Groq's inference speed, we can achieve a >6x speedup from 30 seconds to <5 seconds at much lower cost for causal attribution generation. Additionally, we integrate a Pinecone vector database for Retrieval-Augmented Generation (RAG) on hypophosphatasia papers, and use Cohere reranking to score the importance of each sentence, making it usable for rare disease patients today.

For more details on causal attributions, reference the blog post for more details from the repo this is based off of here [here](#https://gradientscience.org/contextcite/).

## 1. Install the Environment

To set up the environment for this project, you need to install the required dependencies listed in the `requirements.txt` file. You can do this using `pip`. Follow the steps below:

1. **Create a virtual environment (optional but recommended):**
    ```sh
    python -m venv venv
    ```

2. **Activate the virtual environment:**
    - On Windows:
        ```sh
        .\venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source venv/bin/activate
        ```

3. **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## 2. Run the Streamlit App

Once you have installed the required dependencies, you can run the Streamlit app located in `app/app.py`. Use the following command to start the app:

If you can't get it running, focus on app/app.py and context_cite/cc_groq.py.