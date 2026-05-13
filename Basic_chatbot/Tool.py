from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from model import huggingface_model
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests

load_dotenv()

# --- 1. Tools Setup ---

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_number: float, second_number: float, operation: str) -> dict:
    """Perform a basic arithmetic operation: add, sub, mul, div."""
    try:
        if operation == 'add': result = first_number + second_number
        elif operation == 'mul': result = first_number * second_number
        elif operation == 'sub': result = first_number - second_number
        elif operation == 'div':
            if second_number == 0: return {"error": "Division by zero"}
            result = first_number / second_number
        else: return {"error": f"unsupported operation '{operation}'"}
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a symbol (e.g. 'AAPL')."""
    # Fix: Removed spaces in URL and corrected 'apikey' parameter
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=4Q2V77K0DCE9KUJ2"
    r = requests.get(url)
    return r.json()


#MCP Implement
SERVERS = {
    "github": {
        "transport" : "stdio",
        "command" : "/usr/bin/python3",
        "args": [
            "/path/to/github_mcp_server.py"
        ]
    }
}

# List of tools
tools_list = [get_stock_price, search_tool, calculator]



llm = huggingface_model()

llm_with_tool = llm.bind_tools(tools_list)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state['messages']
    response = llm_with_tool.invoke(messages)
    return {'messages': [response]}

# execute tool node - standard name is "tools"
tool_node = ToolNode(tools_list)

# --- 3. Graph Builder ---

def get_graph_builder():
    graph = StateGraph(ChatState)

    # Use "tools" (plural) to satisfy the 'tools_condition' default target
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node) 

    graph.add_edge(START, "chat_node")

    # This condition will now correctly find the "tools" node
    graph.add_conditional_edges("chat_node", tools_condition)
    
    # Loop back to chat_node after tool execution
    graph.add_edge("tools", "chat_node")
    
    return graph