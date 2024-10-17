import streamlit as st
from transformers import pipeline

# Load the question-answering pipeline
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_pipeline = load_qa_model()

# AI context for the chatbot
ai_context = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are designed to think and act like humans. AI can be classified into narrow AI, which is designed to perform a narrow task (e.g. facial recognition, internet searches), and general AI, which can outperform humans at nearly every cognitive task.
Machine learning (ML) is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Deep learning, a subset of machine learning, mimics the human brain in processing data and creating patterns for use in decision making.
Natural language processing (NLP) is a field of AI focused on the interaction between computers and humans through natural language. In essence, it's the way we can enable machines to understand, interpret, and respond to human language.
AI applications include robotics, speech recognition, image analysis, autonomous vehicles, and more.
"""

# Function to generate answers
def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Streamlit UI
st.title("AI Question Answering Chatbot")
st.write("Ask me anything about Artificial Intelligence!")

# Input from user
user_question = st.text_input("Your Question")

if st.button("Get Answer"):
    if user_question:
        response = answer_question(user_question, ai_context)
        st.write(f"**Answer:** {response}")
    else:
        st.write("Please enter a question.")
