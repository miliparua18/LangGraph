from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from model import huggingface_model
import os

load_dotenv()

# Model 
model = huggingface_model()

# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

#Node (STREAMING)
def chat_node(state: ChatState):
    messages = state["messages"]
    #send to llm 
    output = model.invoke(messages)
    return { "messages": [output] }


# Memory
checkpointer = InMemorySaver()

# Graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile(checkpointer=checkpointer)



