from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

class Jokestate(TypedDict):
    topic: str
    jokes : str
    explanation: str


def generate_jokes(state: Jokestate):
    prompt = f"generate a joke on this topic - \n {state['topic']}"
    response = model.invoke(prompt).content
    return {'jokes': response}


def generate_explanation(state: Jokestate):
    prompt = f"generate a explanation on the jokes - \n {state['jokes']}"
    response = model.invoke(prompt).content
    return {'explanation': response}



graph = StateGraph(Jokestate)

graph.add_node('generate_jokes',generate_jokes)
graph.add_node('generate_explanation',generate_explanation)

graph.add_edge(START,'generate_jokes')
graph.add_edge('generate_jokes','generate_explanation')
graph.add_edge('generate_explanation',END)


checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)

config1 = {"configurable": {"thread_id":'1'}}
final_state = workflow.invoke({'topic': 'Pizza'}, config=config1)
print(final_state)

config2 = {"configurable": {"thread_id":'2'}}
final_state = workflow.invoke({'topic': 'Pasta'}, config=config2)
print(final_state)

print(workflow.get_state(config2))