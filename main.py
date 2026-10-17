from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma 

model = ChatGroq(model =  "openai/gpt-oss-20b",temperature=0)


#let user input query 
#retrievers get the content from vectorStore 
#result of retrievers ->  similar chunks 
#similar chunks + user query => a prompt 
#send it to LLM


embedding_model =  HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

vectorStore = Chroma(
    persist_directory= "Chroma-db",
    embedding_function=embedding_model 
)

retriever = vectorStore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k":4,
        "fetch" : 10,
        "lambda_multi": 0.5
    }
)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


#creating prompt template 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """ you are a helpful AI assitant.
         
         use only the provided context to answer to answer the question.
         
         If the answer is not present in the context ,
         say : "I could not find the answer in the document."
         """),
        (
            "human",
            """ Context:{context}

            Question : {question} 
            """
        )
    ]
)

print("RAG system created ")
print("press 0 to exit ")
while True:
    query = input("You : ")
    if query == "0" :
        break
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context" : context,
        "question" : query
    })

    response = llm.invoke(final_prompt)

    print(f"\n AI : {response.content}")

