from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#step 1 loading 
loader = PyPDFLoader("Your PDf File")
documents = loader.load() # reads teh document and returns a list of documents objects with page context and metadata

#step 2 chunking
Splitter = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap=50)
chunks = splitter.split_documents(documents) # takes list of documents and chunks them

#step 3 embedding
embeddings = OpenAIEmbeddings()

#step 4 vectordb
vectordb = chromadb.Chroma(chunks, embeddings)
#step 5 Create Retriever
retriever = vectordb.as_retriver()#default top-k is 4
#step 6 user query
query = "What is the main topic of the document?"
#step 7 retrieve relevant results
results = retriever.get_relevant_documents(query)

#step 8 bulid context for LLM
context = "\n".join([result.page_content for result in results])

#create promptfor LLM
prompt = f"""Based on the following context, answer the question.
 Context:{context} 
 
 Question:{query}
    Answer:"""

#step 9 LLM response
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
response = llm.invoke(prompt)
print(response.content)