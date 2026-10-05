import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# CHECK API KEY
# --------------------------------------------------

if not API_KEY:
    st.error("GEMINI_API_KEY is not found in the .env file.")
    st.info("Create a .env file and add: GEMINI_API_KEY=YOUR_API_KEY")
    st.stop()


# --------------------------------------------------
# GEMINI CLIENT
# --------------------------------------------------

client = genai.Client(api_key=API_KEY)


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# --------------------------------------------------
# CHROMA DATABASE
# --------------------------------------------------

@st.cache_resource
def create_chroma_collection():

    chroma_client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = chroma_client.get_or_create_collection(
        name="study_documents"
    )

    return collection


collection = create_chroma_collection()


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def split_text(text, chunk_size=800, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# READ PDF
# --------------------------------------------------

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# --------------------------------------------------
# ADD DOCUMENT TO CHROMA
# --------------------------------------------------

def add_document_to_database(text, filename):

    chunks = split_text(text)

    if not chunks:
        return 0

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    ).tolist()

    ids = [
        f"{filename}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": filename,
            "chunk": i
        }
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


# --------------------------------------------------
# SEARCH DOCUMENTS
# --------------------------------------------------

def search_documents(question, number_of_results=4):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_results
    )

    documents = results.get("documents", [[]])

    if documents and documents[0]:
        return documents[0]

    return []


# --------------------------------------------------
# ASK GEMINI
# --------------------------------------------------

def ask_gemini(question, context=""):

    if context:

        prompt = f"""
You are an AI Study Assistant.

Answer the student's question using the study material provided below.

Study Material:
{context}

Student Question:
{question}

Instructions:
- Give a clear and correct answer.
- Use simple English.
- Explain difficult concepts in an easy way.
- Use headings or bullet points when useful.
- If the answer is not present in the study material, say that clearly.
"""

    else:

        prompt = f"""
You are an AI Study Assistant.

Student Question:
{question}

Instructions:
- Explain the answer in simple English.
- Make it easy for a college student to understand.
- Use examples when useful.
- Use headings and bullet points when appropriate.
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    return response.text


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("📚 Study Assistant")

    st.write("Upload your study material here.")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file:

        if st.button("📥 Process PDF"):

            with st.spinner("Reading PDF..."):

                pdf_text = extract_pdf_text(
                    uploaded_file
                )

                if not pdf_text.strip():

                    st.error(
                        "No readable text was found in this PDF."
                    )

                else:

                    chunk_count = add_document_to_database(
                        pdf_text,
                        uploaded_file.name
                    )

                    st.success(
                        f"PDF processed successfully! "
                        f"{chunk_count} chunks added."
                    )

    st.divider()

    st.write("### Study Features")

    st.write("💬 Ask questions")
    st.write("📄 PDF question answering")
    st.write("🧠 Simple explanations")
    st.write("📝 Summaries")
    st.write("📌 Important points")


# --------------------------------------------------
# MAIN PAGE
# --------------------------------------------------

st.title("📚 AI Study Assistant Chatbot")

st.write(
    "Ask questions, upload study material, "
    "and learn concepts using AI."
)


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask your study question..."
)


if question:

    # Display user question

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # Find relevant PDF content

    with st.spinner("Thinking..."):

        relevant_documents = search_documents(
            question
        )

        context = "\n\n".join(
            relevant_documents
        )

        try:

            answer = ask_gemini(
                question,
                context
            )

        except Exception as error:

            answer = (
                "Sorry, I could not generate an answer.\n\n"
                f"Error: {error}"
            )


    # Display answer

    with st.chat_message("assistant"):

        st.markdown(answer)


    # Save answer

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )