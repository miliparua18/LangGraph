from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
import os 


load_dotenv()


llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        streaming=True
    )

model = ChatHuggingFace(llm = llm)




def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")


DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # Run ONCE (creates tables)
    checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer)

    # Thread 1 (remembers)
    t1 = {"configurable": {"thread_id": "thread-1"}}
    graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Mili"}]}, t1)
    out1 = graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, t1)
    print("Thread-1:", out1["messages"][-1].content)


