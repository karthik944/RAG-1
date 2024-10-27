import streamlit as st
import subprocess
import re
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from wikipediaapi import Wikipedia
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Setup the Sentence Embeddings Model
model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)

# 2. Setup Wikipedia API
wiki = Wikipedia('RAGBot/Rag-1', 'en')

# 3. Load GPT-2 Model and Tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 4. Function to preprocess the prompt
def preprocess_prompt(prompt: str) -> str:
    prompt = prompt.lower().strip()
    prompt = re.sub(r'[^\w\s]', '', prompt)  # Remove punctuation
    return prompt

# 5. Function to post-process the response
def postprocess_response(response: str) -> str:
    response = re.sub(r'<\|endoftext\|>', '', response)  # Clean end of text token
    return response.strip()

# 6. Function to generate text using local LLaMA model via Ollama
def generate_llama_response(prompt):
    process = subprocess.run(
        ['ollama', 'run', 'llama3.2:latest'],
        input=prompt,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if process.returncode != 0:
        error_message = process.stderr.strip()
        return f"Error running Ollama: {error_message}"
    
    response = process.stdout.strip()
    return response

# 7. Function to generate text using GPT-2
def generate_gpt2_response(prompt):
    # Encode the input prompt
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    
    # Set maximum input length and new tokens to generate
    max_input_length = 512  # or another suitable length
    max_new_tokens = 50  # Adjust this based on how much output you want

    # Truncate input if it exceeds the max input length
    if inputs.size(1) > max_input_length:
        inputs = inputs[:, :max_input_length]

    # Generate the response
    outputs = gpt2_model.generate(inputs, max_new_tokens=max_new_tokens, num_return_sequences=1)
    
    # Decode and return the generated response
    response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 8. Function to search Wikipedia
def search_wikipedia(query):
    stop_words = {"what", "who", "how", "is", "was", "are", "which", "when", "where", "why", "did", "do", "does"}
    words = [word for word in query.split() if word.lower() not in stop_words]
    
    paragraphs_list = []

    for word in words:
        doc = wiki.page(word).text
        if doc:
            paragraphs = doc.split('\n\n')
            paragraphs_list.extend(paragraphs)

    if not paragraphs_list:
        return None
    
    docs_embed = model.encode(paragraphs_list, normalize_embeddings=True)
    query_embed = model.encode(query, normalize_embeddings=True)
    similarities = np.dot(docs_embed, query_embed.T)
    top_3_idx = np.argsort(similarities, axis=0)[-3:][::-1].tolist()
    most_similar_documents = [paragraphs_list[idx] for idx in top_3_idx]
    
    context = "\n\n".join(most_similar_documents)
    return context

# 9. Combining retrieval and generation (RAG workflow)
def rag_response(query, model_choice='llama'):
    start_time = time.time()  # Start timing

    context = search_wikipedia(query)

    if context is None:
        return "I couldn't find relevant information from Wikipedia."

    prompt = f"""
    I have some information that might be relevant to answering a question.
    Here's the information:

    {context}

    Based on this information, please answer the following question:

    {query}

    If the information provided doesn't contain the answer, please say "I don't have enough information to answer this question."
    """

    if model_choice == 'LLaMA with RAG':
        response = generate_llama_response(prompt)
    elif model_choice == 'GPT-2 with RAG':
        response = generate_gpt2_response(prompt)
    else:
        # Direct model usage without RAG
        if model_choice == 'LLaMA':
            response = generate_llama_response(query)  # Direct usage without RAG
        else:  # GPT-2
            response = generate_gpt2_response(query)

    end_time = time.time()  # End timing
    response_time = end_time - start_time

    return response, response_time

# 10. Streamlit app interface
st.title("RAG-Powered AI Assistant with LLaMA3 and GPT-2 Integration")

# Create a session state to store chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Tabs for Chat and Conversation History
tab1, tab2 = st.tabs(["Chat", "Conversation History"])

# Chat Tab
with tab1:
    query = st.text_input("Your Query:", key="query_input")
    
    # Select Model
    model_choice = st.selectbox("Select Model:", ["LLaMA with RAG", "GPT-2 with RAG", "LLaMA", "GPT-2"])

    if st.button("Submit"):
        if query:
            with st.spinner("Fetching response..."):
                # Preprocess the user input
                preprocessed_input = preprocess_prompt(query)
                response, response_time = rag_response(preprocessed_input, model_choice=model_choice)

                # Post-process the response
                cleaned_output = postprocess_response(response)

                # Store the conversation in chat history
                st.session_state.chat_history.append({
                    "user": query,
                    "chatbot": cleaned_output,
                    "model": model_choice,  # Add the selected model to chat history
                    "response_time": response_time  # Store the response time
                })

                st.subheader("Response:")
                st.write(cleaned_output)

                # Display the response time
                st.write(f"Response Time: {response_time:.2f} seconds")
        else:
            st.warning("Please enter a query to get a response.")

# Conversation History Tab
with tab2:
    st.write("### Conversation History")
    if st.session_state.chat_history:
        for i, chat in enumerate(st.session_state.chat_history):
            st.write(f"**User:** {chat['user']}")
            st.write(f"**Chatbot:** {chat['chatbot']}")
            st.write(f"**Model Used:** {chat['model']}")
            st.write(f"**Response Time:** {chat['response_time']:.2f} seconds")  # Display response time
    else:
        st.write("No conversation history available.")
