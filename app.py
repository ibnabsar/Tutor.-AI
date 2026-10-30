import os
import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --- Page Configuration ---
st.set_page_config(page_title="AI Custom Tutor Prototype", page_icon="📚", layout="centered")

st.title("📚 AI Study Companion & Scenario Tutor")
st.write("Upload your notes (PDF or TXT), and chat with an AI that teaches you using real-life examples and scenarios!")

# --- API Key Input ---
api_key = st.text_input("Enter your Google Gemini API Key:", type="password")

if not api_key:
    st.warning("Please enter your Google Gemini API key to proceed. (You can get a free one from Google AI Studio).")
    st.stop()

# Set environment variable for LangChain Google integration
os.environ["GOOGLE_API_KEY"] = api_key

# --- Sidebar for Uploading Notes ---
st.sidebar.header("📁 Step 1: Upload Study Material")
uploaded_file = st.sidebar.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

# Persistent directory for local vector storage
persist_directory = "./chroma_db"

if uploaded_file is not None:
    with st.spinner("Processing and indexing your notes..."):
        # Save uploaded file temporarily
        temp_file_path = f"./temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load document based on file type
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
        else:
            loader = TextLoader(temp_file_path)
        
        docs = loader.load()

        # Split text into bite-sized chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        # Create Embeddings and Vector Store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=persist_directory
        )

        # Clean up temp file
        os.remove(temp_file_path)
        st.sidebar.success("Notes successfully uploaded and indexed!")

# --- Main Chat Interface ---
if os.path.exists(persist_directory):
    # Load existing vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Initialize Gemini Model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

    # Custom Pedagogical System Prompt (The Secret Sauce)
    system_prompt = (
        "You are an expert, highly empathetic, and engaging personal tutor. "
        "Your task is to teach the student using **only** the provided context extracted from their uploaded notes.\n\n"
        "Guidelines:\n"
        "1. If a student asks a question or expresses confusion, do not just give a dry textbook definition.\n"
        "2. Always break down complex ideas into simple components.\n"
        "3. Invent a relatable, real-life scenario, analogy, or case study (e.g., from daily life, business, or sports) that fits the situation to make it easy to digest.\n"
        "4. End your response with a quick checking question to ensure the student understood the concept.\n"
        "5. If the answer cannot be found in the context, politely state: 'I couldn't find that in your uploaded notes.'\n\n"
        "Context from notes:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Build RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Initialize chat history state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display prior chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if user_query := st.chat_input("Ask a question about your notes or ask for a scenario explanation..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response from AI
        with st.chat_message("assistant"):
            with st.spinner("Thinking of a real-life scenario..."):
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Please upload your study notes using the sidebar to start chatting with your AI tutor.")
