from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import time
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

class Crashstate(TypedDict):
    input: str
    step1: str
    step2: str
    step3: str

def step_1(state: Crashstate)->Crashstate:
    print("step1 executed")
    return {"step1":"done","input":state["input"]}

def step_2(state: Crashstate)->Crashstate:
    print("step2 hanging........now manually interrupt from the notebook toolbar(STOP button)")
    time.sleep(30) #simmulate long running hang
    return {"step2":"done"}

def step_3(state: Crashstate) -> Crashstate:
    print("step3 executed")
    return {"step3":"done"}


graph = StateGraph(Crashstate)

graph.add_node('step_1', step_1)
graph.add_node('step_2',step_2)
graph.add_node('step_3',step_3)


graph.add_edge(START,'step_1')
graph.add_edge('step_1','step_2')
graph.add_edge('step_2','step_3')
graph.add_edge('step_3',END)


checkpointer = InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)

try:
    print("Running Graph: please manually interrupt during step 2...."),
    workflow.invoke({"input":"start"}, config={"configurable": {"thread_id": 3}})


except KeyboardInterrupt:
    print("karnel manually interrupt (crash simulated)")


workflow.get_state({"configurable": {"thread_id": 3}})


final_state = workflow.invoke(None, config={"configurable": {"thread_id": 3}})
print("final_state:", final_state)

workflow.get_state({"configurable": {"thread_id": 3}})

