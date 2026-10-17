
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter 

from langchain_community.document_loaders import PyPDFLoader
data = PyPDFLoader("demonstrations/SQLnotes.pdf")

docs = data.load()

"""splitter = TokenTextSplitter(
    chunk_size = 100,
    chunk_overlap = 10
)""" # for TokenTextSplitter - comes under token based splitting - TokenTextSplitter focuses on the token limit.

splitter = RecursiveCharacterTextSplitter( #comes under chacracter based slpitting - RecursiveCharacterTextSplitter uses separators to find sensible places to split, tries to keep meaningful pieces of text together, splitting at suitable separators where possible.
    chunk_size = 100,
    chunk_overlap = 10
)#looks for seperators like - ["\n\n", "\n", " ", ""]

chunks = splitter.split_documents(docs)
print(len(chunks))
print(chunks[0].page_content)

