import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


dotenv_path = os.path.join("G:\\Pycharm_directory\\TRANSACTION_RAG\\.venv\\.env")  
load_dotenv(dotenv_path=dotenv_path)
api_key = os.getenv("OPENAI_API_KEY")
DB_FAISS_PATH = os.path.join(os.getcwd(), "faiss_index")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    # Create FAISS index from documents
    db = FAISS.from_texts(text_chunks, embeddings)
    #db = FAISS.from_documents(text_chunks, embeddings)
    # Save the FAISS index locally
    db.save_local(DB_FAISS_PATH)
    print(f"FAISS index saved to {DB_FAISS_PATH}")
    st.success(f"FAISS index saved to {DB_FAISS_PATH}")

# def get_vector_store(text_chunks):
#     embeddings = OpenAIEmbeddings(openai_api_key=api_key)
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """

    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "answer is not available in the context" — don't provide a wrong answer.
    1. You are an AI assistant who can be helping in the Transaction classification system 
    2. You need to be very conscious about revealing the personal data of the account holder
    3. You should not provide any irrelevant contents based on the query 
    4. You should be very professional
    5. Always give the elaborated answer with clear explanation
    6. Always display the transaction details in a tabular format

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """

    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # or "gpt-4"
        temperature=0.3,
        openai_api_key=api_key
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config(page_title="AI-ASSISTANT")
    st.header("TRANSACTION CLASSIFIER - RAG APPROACH ", divider='rainbow')

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()

