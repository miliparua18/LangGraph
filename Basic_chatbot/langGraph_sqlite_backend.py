from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from model import huggingface_model
import os
import sqlite3
from Tool import get_graph_builder

load_dotenv()

# Model 

model = huggingface_model()

# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


#create database

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# Memory
checkpointer = SqliteSaver(conn)
builder = get_graph_builder()

workflow = builder.compile(checkpointer=checkpointer)



#optimized thread retrieval
def retrieve_all_threads() -> List[str]:
    """
    Directly queries the SQLite database for unique thread IDs.
    Much more efficient than checkpointer.list() for large databases.
    """
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'")
        if not cur.fetchone():
            return []
            
        cur.execute("SELECT DISTINCT thread_id FROM checkpoints")
        return [row[0] for row in cur.fetchall()]
    except Exception as e:
        print(f"Error retrieving threads: {e}")
        return []