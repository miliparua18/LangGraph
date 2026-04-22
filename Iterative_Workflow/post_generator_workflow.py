from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

#State
class PostState(TypedDict):
    topic: str
    post: str
    quality: str
    iteration: int

#Node 1: Generate post
def generate_post(state: PostState):
    messages = [
        SystemMessage(content="You are a professional LinkedIn content writer."),
        HumanMessage(content=f"Write a short engaging LinkedIn post on: {state['topic']}")
    ]

    result = model.invoke(messages)

    return {
        'post': result.content,
        'iteration': 0
    }

#Node 2: Improve post
def improve_post(state: PostState):
    messages = [
        SystemMessage(content="You improve posts to make them more engaging and high quality."),
        HumanMessage(content=f"Improve this post:\n{state['post']}")
    ]

    result = model.invoke(messages)

    return {
        'post': result.content,
        'iteration': state['iteration'] + 1
    }

#Node 3: Evaluate quality
def evaluate_post(state: PostState):
    messages = [
        SystemMessage(content="You evaluate content quality."),
        HumanMessage(content=f"Is this post high quality? Reply only 'good' or 'bad'.\n{state['post']}")
    ]

    result = model.invoke(messages)
    quality = result.content.lower()

    if "good" in quality:
        quality = "good"
    else:
        quality = "bad"

    return {'quality': quality}

#Condition (loop)
def route(state: PostState):
    if state['quality'] == "good" or state['iteration'] >= 3:
        return END
    else:
        return "improve_post"

#Graph
graph = StateGraph(PostState)

graph.add_node('generate_post', generate_post)
graph.add_node('improve_post', improve_post)
graph.add_node('evaluate_post', evaluate_post)

graph.add_edge(START, 'generate_post')
graph.add_edge('generate_post', 'evaluate_post')

graph.add_conditional_edges('evaluate_post', route)

graph.add_edge('improve_post', 'evaluate_post')

workflow = graph.compile()

#Input
initial_state = {
    'topic': 'Benefits of AI in daily life'
}

#Run
result = workflow.invoke(initial_state)

print(result['post'])