#load the pdf 
#split into chunks
#create embeddings 
#store into chroma
 
#all these for main RAG project - CourseMateAI
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma 
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("document_loaders/deeplearning.pdf")
docs = data.load()

# do the chunking of data then store those chunks in vector db
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)


#embedding_model = HuggingFaceEmbeddings(
 #   model_name="sentence-transformers/all-MiniLM-L6-v2"
#)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="Chroma-db"
)
