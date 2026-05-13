from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from model import huggingface_model
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# --- 1. Tools Setup ---

search_tool = DuckDuckGoSearchRun(region="us-en")


#MCP Implement
Servers = MultiServerMCPClient({
    "github": {
        "transport" : "stdio",
        "command" : "/usr/bin/python3",
        "args": ["/path/to/github_mcp_server.py"]
    }
}
)






llm = huggingface_model()



class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph():
    tools = await Servers.get_tools()
    
    llm_with_tool = llm.bind_tools(tools)
    
    async def chat_node(state: ChatState):
        """LLM node that may answer or request a tool call."""
        messages = state['messages']
        response = await llm_with_tool.ainvoke(messages)
        return {'messages': [response]}
    
    tool_node = ToolNode(tools)
    graph = StateGraph(ChatState)

    
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node) 

    graph.add_edge(START, "chat_node")

    # This condition will now correctly find the "tools" node
    graph.add_conditional_edges("chat_node", tools_condition)
    
    # Loop back to chat_node after tool execution
    graph.add_edge("tools", "chat_node")
    
    return graph.compile()
    



async def main():
    chatbot = await build_graph()
    result = await chatbot.ainvoke({'messages': [HumanMessage(content="Find the modulus of 12345 and 23 and give answer like a cricket commentator.")]})
    print(result['messages'][-1].content)


if __name__ == '__main__':
    asyncio.run(main())