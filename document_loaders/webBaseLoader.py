from langchain_community.document_loaders import   WebBaseLoader

url  = "https://en.wikipedia.org/wiki/List_of_common_misconceptions"

data = WebBaseLoader(url) #creates an object
docs = data.load()

print(len(docs))

print(docs[0].page_content)