
# Integrating MCP-Style Architecture for Langchain LangGraph Workflow

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
# Tool Layer (MCP idea)
# ========================
class ToolManager:
    def __init__(self):
        self.tools = {}
    def register_tool(self, name, func): 
        self.tools[name] = func
    def call_tool(self, name, *args): #*args means accept any number of positional arguments
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")
        return self.tools[name](*args)#find the tool by name and call it with provided arguments


# ========================
#  Define Tool
# ========================
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


def route_tool(state: GraphState):

    if state["next_tool"] == "finish":
        return "end"

    return state["next_tool"]



# ===== Create Tool Manager Instance
tool_manager = ToolManager()

# ===== Register tool
tool_manager.register_tool("retrieve", retrieve_tool)
tool_manager.register_tool("summarize", summarize_tool)
tool_manager.register_tool("evaluate", evaluate_tool)


# ========================
# 2. State
# ========================

class GraphState(TypedDict):
    query: str # The user’s question
    docs : List[str] #Stores retrieved documents, filled after retrieval
    answer: str # Holds the generated response, filled after summarization
    next_tool : str #Holds the name of the next tool to call, filled by the tool_node

# ========================
# 3. Node functions
# ========================

def tool_node(state: GraphState):
    query = state["query"]
    docs = state.get("docs", [])
    answer = state.get("answer", "")

    prompt = f"""
    You are an AI system with access to these tools:
    - retrieve(query)
    - summarize(query, docs)
    - evaluate(answer)

    Decide which tool to use next.

    Rules:
    - If no docs exist → use retrieve
    - If docs exist but no answer → use summarize
    - If answer exists but may need improvement → use evaluate
    - If the answer is complete and good → use finish

    Respond with ONLY one of:
    retrieve OR summarize OR evaluate OR finish
    """

    decision = llm.invoke(prompt).content.strip().lower()

    return {"next_tool": decision}



def retrieve_node(state: GraphState):
    #1.read 
    query = state['query']
    #2. process
    docs = tool_manager.call_tool("retrieve",query)
    #3. write
    return {'docs': docs}

def summarize_node(state: GraphState): # (Augumention + Generation) in RAG
    #1. read 
    query = state['query']
    docs = state.get("docs", []) #get() avoids crashes if docs don’t exist
    #2. process - mock summarization
    
    answer = tool_manager.call_tool("summarize", query, docs)
    #3. write   
    return {'answer': answer}

def evaluate_node(state: GraphState):
    answer = state['answer']

    evaluated_answer = tool_manager.call_tool("evaluate", answer)
    
    return {"answer": evaluated_answer} #No updates to the state

# ========================
# 4. Graph building
# ========================
builder = StateGraph(GraphState)

# ===== Add nodes
# You are registering name, function builder.add_node("name", function)
builder.add_node("tool_selector", tool_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("summarize", summarize_node)
builder.add_node("evaluate", evaluate_node)

#Set an entry point
builder.set_entry_point("tool_selector") #start workflow at the decide node

# ===== Add Conditional Routing
builder.add_conditional_edges("tool_selector", route_tool)

#===== Add Direct Edges
builder.add_edge("retrieve", "tool_selector")#after retrieval, always summarize
builder.add_edge("summarize", "tool_selector")#after summarization, always evaluate
builder.add_edge("evaluate", "tool_selector")#after evaluation, always decide next tool

#===== Set Finish point
builder.set_finish_point("end") #After evaluate, stop the workflow

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
    "next_tool": ""
})


print("\nFinal Answer:\n")
print(result['answer'])










