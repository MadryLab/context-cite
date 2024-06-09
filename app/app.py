import streamlit as st
from context_cite.cc_groq import GroqContextCiter
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from openai import OpenAI
import torch as ch
from typing import List
from difflib import get_close_matches
from nltk.tokenize import sent_tokenize

load_dotenv()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

openai_client =  OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

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
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# Connect to the Pinecone index
index = pc.Index("stanfordrdhack") #index generated from other hackathon project for simplicity

def perform_rag(query):
    # query_embedding = _get_embedding(query)
    # Perform a similarity search in the Pinecone index
    # search_results = index.query(vector=query_embedding, namespace='hpp', include_values=True,
    # include_metadata=True, top_k=5)

    # print(search_results)
    # Extract the most relevant contexts
    # relevant_contexts = [match['metadata']['text'] for match in search_results['matches']]

    # Combine the relevant contexts
    # combined_context = " ".join(relevant_contexts)
    combined_context = "Hypophosphatasia is a rare, inherited metabolic disorder that affects the development of bones and teeth. It is caused by mutations in the ALPL gene, which encodes an enzyme called alkaline phosphatase. People with hypophosphatasia have low levels of alkaline phosphatase, which leads to abnormal mineralization of bones and teeth. The severity of the condition can vary widely, from mild forms that only affect the teeth to severe forms that can be life-threatening. Treatment for hypophosphatasia is focused on managing symptoms and preventing complications. This may include medications to increase alkaline phosphatase levels, physical therapy, and surgery to correct bone deformities."
    return combined_context

if "response" not in st.session_state:
    st.session_state.response = None

# Accept user input
if prompt := st.chat_input("Ask a question about hypophosphatasia:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.response is None:
        # Define the context
        context = perform_rag(prompt)

        # Initialize the GroqContextCiter
        cc = GroqContextCiter(groq_model='llama3-70b-8192', context=context, query=prompt, num_ablations=8)
        if "cc" not in st.session_state:
            st.session_state.cc = cc
        # Get the response from the model
        response = cc.response
        st.session_state.response = response
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    if prompt.startswith("\\cite") and st.session_state.cc is not None:
        # Assuming 'results' is already defined as per the instructions
        sentence = prompt.split("\\cite")[1].strip()
        response = st.session_state.response
        answer_sentences = sent_tokenize(response)
        cc = st.session_state.cc
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
