from langchain_community.document_loaders import PyPDFDirectoryLoader
import os

DATA_PATH = "data"



# Ingesting the document

def load_documents():
    # Create directory if it doesn't exist
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
     
    doc_loader = PyPDFDirectoryLoader(DATA_PATH)
    return doc_loader.load()




# try:
#     docs = load_documents()
#     if docs:
#         print(docs[0])
#     else:
#         print("No documents were found in the directory. Please add PDF files to the 'data' directory.")
# except Exception as e:
#     print(f"Error loading documents: {e}")