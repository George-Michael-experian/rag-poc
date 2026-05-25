
# 1. Imports
# 2. State definition
# 3. Node functions
# 4. Graph building
# 5. Run code

# ========================
# 1. Imports
# ========================
from typing import TypedDict, List # TypedDict It lets you define a dictionary with a fixed structure.
from langgraph.graph import StateGraph #StateGraph is the engine that runs your workflow # A flowchart manager
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


# ========================
# 2. State
# ========================

class GraphState(TypedDict):
    query: str # The user’s question
    docs : List[str] #Stores retrieved documents, filled after retrieval
    answer: str # Holds the generated response, filled after summarization
    use_rag : bool #Decision from the “decide node” on whether to use RAG or not

# ========================
# 3. Node functions
# ========================

def decide_node(state: GraphState):
    
    query = state["query"]

    prompt = f"""
    Determine if this question requires retrieving external documents.

    Question: {query}

    Answer ONLY with True or False.
    """

    decision = llm.invoke(prompt).content.strip().lower() 

    use_rag = "true" in decision

    return{"use_rag": use_rag}



def retrieve_node(state: GraphState):
    #1.read 
    query = state['query']
    #2. process
    docs = retriever.invoke(query) #retrieves relevant documents based on the query
    docs = [doc.page_content for doc in docs] #extracts text content from retrieved documents
    #3. write
    return {'docs': docs}

def summarize_node(state: GraphState): # (Augumention + Generation) in RAG
    #1. read 
    query = state['query']
    docs = state.get("docs", []) #get() avoids crashes if docs don’t exist
    #2. process - mock summarization
    context = "\n".join(docs)
    
    prompt = f"""
    Answer the question using the context below.

    Question: {query}
    Context: {context}

    Provide a clear and helpful answer.
    """
    
    answer = llm.invoke(prompt)
    #3. write   
    return {'answer': answer.content}

def evaluate_node(state: GraphState):
    answer = state['answer']
    if len(answer) < 50:
        return {"answer" : answer  +  " (needs improvement)"}
    
    return {} #No updates to the state

# ========================
# 4. Graph building
# ========================
builder = StateGraph(GraphState)

# ===== Add nodes
# You are registering name, function builder.add_node("name", function)
builder.add_node("decide", decide_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("summarize", summarize_node)
builder.add_node("evaluate", evaluate_node)

#Set an entry point
builder.set_entry_point("decide") #start workflow at the decide node

# ===== Add Conditional Routing
def route_decision(state: GraphState):
    if state['use_rag']:
        return "retrieve"
    else:
        return "summarize"

"""connects decide node to either retrieve or summarize 
based on the route_decision function """   

builder.add_conditional_edges("decide", route_decision)

#===== Add Direct Edges
builder.add_edge("retrieve", "summarize")#after retrieval, always summarize
builder.add_edge("summarize", "evaluate")#after summarization, always evaluate

#===== Set Finish point
builder.set_finish_point("evaluate") #After evaluate, stop the workflow

#===== Compile the Graph
graph = builder.compile() #Converts your design → executable system

# ========================
# 5. Run code
# ========================
query = input("Enter your query: ")

result = graph.invoke({
    "query": query,
    "docs": [],
    "answer": "",
    "use_rag": False
})


print("\nFinal Answer:\n")
print(result['answer'])










