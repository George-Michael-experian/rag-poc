import requests # Allows Python to send HTTP requests (like a browser, but in code)
from langgraph.graph import StateGraph #StateGraph is the engine that runs your workflow # A flowchart manager
from typing import TypedDict, List # TypedDict It lets you define a dictionary with a fixed structure.
# ========= Build the MCP Client class
class MCPClient:
    def __init__(self, server_url):
      self.server_url = server_url
    
    def call_tool(self, tool_name, **kwargs):
        response = requests.post(
            f"{self.server_url}/tool",
            json={
                "tool": tool_name,
                "args": kwargs #**kwargs means take all the keyword arguments passed to this function and put them in a dictionary called args
            }
        )

        return response.json()["result"]

""" Create an instance of the MCPClient, pointing to our server URL 
(where the MCP server is running) """  

mcp_client = MCPClient("http://localhost:8000") 

# ========================
#  State
# ========================

class GraphState(TypedDict):
    query: str # The user’s question
    docs : List[str] #Stores retrieved documents, filled after retrieval
    answer: str # Holds the generated response, filled after summarization
    next_tool : str #Holds the name of the next tool to call, filled by the tool_node


def retrieve_node(state: GraphState):
    #1.read 
    query = state['query']
    #2. process
    docs = mcp_client.call_tool("retrieve",query=query)
    #3. write
    return {'docs': docs}

def summarize_node(state: GraphState): # (Augumention + Generation) in RAG
    #1. read 
    query = state['query']
    docs = state.get("docs", []) #get() avoids crashes if docs don’t exist
    #2. process - mock summarization
    
    answer = mcp_client.call_tool("summarize", query=query, docs=docs)
    #3. write   
    return {'answer': answer}

def evaluate_node(state: GraphState):
    answer = state['answer']

    evaluated_answer = mcp_client.call_tool("evaluate", answer=answer)
    
    return {"answer": evaluated_answer} #No updates to the state
