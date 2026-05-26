from fastapi import FastAPI # A framework to build APIs (servers that accept requests)
from pydantic import BaseModel # This defines what input looks like.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# ========================
# LLM + RAG Setup
# ========================

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Load PDF
loader = PyPDFLoader("your_file.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Create vector DB
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(chunks, embeddings)

# Retriever (tool)
retriever = vectordb.as_retriever()


app = FastAPI() #create API server instance

#This is how MCP works conceptually
# ===== Request Schema
class ToolRequest(BaseModel): #When someone calls your server, they must send tool and input
    tool: str #which function to run
    args: dict #inputs to that function

# ===== TOOLS

def retrieve_tool(query):
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]

def summarize_tool(query, docs): # (Augumention + Generation) in RAG
   
    #2. process - mock summarization
    context = "\n".join(docs)
    
    prompt = f"""
    Answer the question using the context below.

    Question: {query}
    Context: {context}

    Provide a clear and helpful answer.
    """
    
    response = llm.invoke(prompt)
    #3. write   
    return response.content

def evaluate_tool(answer):
    if len(answer) < 50:
        return answer  +  " (needs improvement)"
    
    return answer


#Tool Registry

TOOLS = {
    "retrieve": retrieve_tool,
    "summarize": summarize_tool,
    "evaluate": evaluate_tool,
}


@app.post("/tool") #When someone sends a POST request to /tool, run this function
# "/tool" is the endpoint (URL) where your server listens for requests.
# POST request -> placing an order

def call_tool(request: ToolRequest):
    tool_name = request.tool
    args = request.args

    if tool_name not in TOOLS:
        return {"error": "Tool not found"}

    result = TOOLS[tool_name](**args)
    return {"result": result}

