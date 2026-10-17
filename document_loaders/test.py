#phase1 -) loading notes/documents from users 
#there are various ways to do it 
from langchain_community.document_loaders import TextLoader
data = TextLoader("document_loaders/notes.txt") #creates an object

docs = data.load()#contains a list  [meta data(like source) and page content]

#print(docs) -) prints the list 
print(docs[0])

