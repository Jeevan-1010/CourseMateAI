from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document_loaders/SQLnotes.pdf")
docs = data.load()

print(len(docs)) # contains multiple docs(represents pages in the pdf), each doc conatining a list of metadta and page_content 
#now if u try to feed this docs to an AI, then this docs exceeds the context window of the AI , so it can't take all at once