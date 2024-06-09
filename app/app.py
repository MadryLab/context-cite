import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from context_cite.cc_groq import GroqContextCiter
from pinecone import Pinecone
from openai import OpenAI
from groq import Groq
import cohere
import torch as ch
from typing import List
from difflib import get_close_matches
from nltk.tokenize import sent_tokenize

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]

openai_client =  OpenAI(
    api_key=OPENAI_API_KEY,
)

groq_client = Groq(api_key=GROQ_API_KEY)

cohere_client = cohere.Client(api_key=COHERE_API_KEY)
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# Set the title of the Streamlit app
st.title("Hypophosphatasia Q&A with Citations")

# Set the subtitle of the Streamlit app
st.subheader("To know which context sentences contributed to a given answer, use \\cite `your_sentence_here`.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def _get_embedding(text) -> List[float]:
    if isinstance(text, str):
        text = [text]
    embedding_response = openai_client.embeddings.create(input=text, 
                                                         model="text-embedding-3-small",
                                                         dimensions=256) #hardcoded for our index
    embeddings = ch.stack([ch.tensor(item.embedding) for item in embedding_response.data])
    if embeddings.dim() == 1:
        return embeddings.unsqueeze(0)
    return embeddings.squeeze().tolist()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
# Connect to the Pinecone index
index = pc.Index("stanfordrdhack") #index generated from other hackathon project for simplicity

def perform_rag(query):
    query_embedding = _get_embedding(query)
    # Perform a similarity search in the Pinecone index
    search_results = index.query(
        vector=query_embedding,
        namespace='hpp',
        include_values=True,
        include_metadata=True,
        top_k=5
    )

    # Extract the most relevant contexts
    relevant_contexts = [match['metadata']['text'] for match in search_results['matches']]

    # Combine the relevant contexts
    combined_context = " ".join(relevant_contexts)
    combined_context += "Hypophosphatasia is a rare, inherited metabolic disorder that affects the development of bones and teeth. It is caused by mutations in the ALPL gene, which encodes an enzyme called alkaline phosphatase. People with hypophosphatasia have low levels of alkaline phosphatase, which leads to abnormal mineralization of bones and teeth. The severity of the condition can vary widely, from mild forms that only affect the teeth to severe forms that can be life-threatening. Treatment for hypophosphatasia is focused on managing symptoms and preventing complications. This may include medications to increase alkaline phosphatase levels, physical therapy, and surgery to correct bone deformities."
    return combined_context

# Accept user input
if prompt := st.chat_input("Ask a question about hypophosphatasia:"):
    # Add user message to chat history
    user_query_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_query_message)
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    if prompt.startswith("\\cite") and st.session_state.cc is not None:
        # Assuming 'results' is already defined as per the instructions
        cc = st.session_state.cc
        sentence = prompt.split("\\cite")[1].strip()
        answer_sentences = sent_tokenize(cc.response)
        # Fuzzily find the closest sentence in answer_sentences
        closest_sentence = get_close_matches(sentence, answer_sentences, n=1, cutoff=0.0)
        
        if closest_sentence:
            closest_sentence = closest_sentence[0]
        else:
            closest_sentence = "No close match found."

        st.write(f"Closest sentence: {closest_sentence}")
        attr_df = cc.get_attributions(closest_sentence, as_dataframe=True, top_k=5).data
        attr_df = attr_df[attr_df['Score'] != 0]
        with st.chat_message("assistant"):
            st.write(attr_df)
    else:
        # Define the context
        context = perform_rag(prompt)

        # Initialize the GroqContextCiter
        cc = GroqContextCiter(
            groq_model='llama3-70b-8192',
            context=context,
            query=prompt,
            groq_client=groq_client,
            openai_client=openai_client,
            cohere_client=cohere_client,
            num_ablations=8
        )
        st.session_state.cc = cc
        response = cc.response
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        cc.messages.append(user_query_message)
        # Add assistant response to chat history
        assistant_message = {"role": "assistant", "content": response}
        st.session_state.messages.append(assistant_message)
        cc.messages.append(assistant_message)
