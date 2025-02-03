import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import (SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate,ChatPromptTemplate)

import streamlit as st

# Custom CSS for styling
st.markdown("""
    <style>
        /* Main container styling */
        .stApp {
            background-color: #f5f5f5;
            font-family: 'Arial', sans-serif;
        }

        /* Header styling */
        .stMarkdown h1 {
            color: #2c3e50;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }

        /* Subheader styling */
        .stMarkdown h2 {
            color: #34495e;
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }

        /* Body text styling */
        .stMarkdown p {
            color: #2c3e50;
            font-size: 1.1rem;
            line-height: 1.6;
            margin-bottom: 1rem;
        }

        /* Button styling */
        .stButton button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            border: none;
            cursor: pointer;
        }

        .stButton button:hover {
            background-color: #2980b9;
        }

        /* Chat bubble styling */
        .chat-bubble {
            background-color: #ecf0f1;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            max-width: 70%;
        }

        .chat-bubble.user {
            background-color: #3498db;
            color: white;
            margin-left: auto;
        }

        .chat-bubble.bot {
            background-color: #ecf0f1;
            color: #2c3e50;
            margin-right: auto;
        }

        /* Input box styling */
        .stTextInput input {
            border-radius: 5px;
            padding: 0.5rem;
            font-size: 1rem;
            border: 1px solid #bdc3c7;
        }

        .stTextInput input:focus {
            border-color: #3498db;
            outline: none;
        }

        /* Footer styling */
        .footer {
            text-align: center;
            padding: 1rem;
            background-color: #2c3e50;
            color: white;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("# Welcome to DeepSeek Chatbot with Ollama")

# Subheader
st.caption("## Your AI-Powered Assistant")

# sidebar config

with st.sidebar:
    st.header("Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:7b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilites")
    st.markdown("""
        - Python Expert
        - Debugging Assistent
        - Code Documentation
        - Solution Design
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [Langchain](https://python.langchain.com/)")


# initiate the chat engine 


llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions"
    "with strategic print statements for debugging. Always respond in English"
)


if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role":"ai","content":"Hi I'm Deepseek. How can I help you to code"}]

chat_container = st.container()

with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


#chat input and processing 
user_query = st.chat_input("Type your coding question here...")

def generate_ai_response(prompt_chain):
    processing_pipeline= prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})


def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # generate AI response
    with st.spinner("Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
        
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    st.rerun()