from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import os
import sqlite3

load_dotenv()

# Model 
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    streaming=True
)

model = ChatHuggingFace(llm=llm)

# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

#Node (STREAMING)
def chat_node(state: ChatState):
    messages = state["messages"]
    #send to llm 
    output = model.invoke(messages)
    return { "messages": [output] }

#create database

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# Memory
checkpointer = SqliteSaver(conn)

# Graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)