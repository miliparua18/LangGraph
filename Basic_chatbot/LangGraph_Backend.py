from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os


load_dotenv()

#Load Model
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

#State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

#Node
def chat_node(state: ChatState):
    messages = state["messages"]

    # Call model
    output = model.invoke(messages)

    # Return response
    return {
        "messages": [output]
    }

#Memory
checkpointer = MemorySaver()

#Graph
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

workflow = graph.compile(checkpointer=checkpointer)

#Thread ID (conversation memory)
thread_id = "1"
config = {"configurable": {"thread_id": thread_id}}

#First input
initial_state = {
    "messages": [HumanMessage(content="What is the capital of India?")]
}

result = workflow.invoke(initial_state, config=config)
print("AI:", result['messages'][-1].content)
