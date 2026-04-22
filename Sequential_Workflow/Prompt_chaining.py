from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict
import os


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm = llm)


class BlogStates(TypedDict):
    topic: str
    outline: str
    content: str


def create_outline(state: BlogStates) -> BlogStates:
    # extract topic
    topic = state['topic']

    # call llm and generate outline
    prompt = f"Generate a detailed outline for a blog on this topic - {topic}"
    outline = model.invoke(prompt)

    # update outline in the state

    state['outline'] = outline

    return state



def create_blog(state: BlogStates) -> BlogStates:
    topic = state['topic']
    outline = state['outline']

    prompt = f"Write a detailed blog on the topic {topic} using the following outline /n {outline}"

    content = model.invoke(prompt).content

    state['content'] = content

    return state


def evaluate_rate(state: BlogStates)->BlogStates:
    outline = state['outline']
    content = state['content']

    prompt = f"Based on this outline rate my blog and generate a integer score {outline}"
    outline = model.invoke(prompt)
    state['outline'] = outline

    return state

graph = StateGraph(BlogStates)

graph.add_node('create_outline',create_outline)
graph.add_node('create_blog',create_blog)
graph.add_node('evaluate_rate',evaluate_rate)


graph.add_edge(START,'create_outline')
graph.add_edge('create_outline', 'create_blog')
graph.add_edge('create_blog','evaluate_rate')
graph.add_edge('evaluate_rate', END)

result = model.invoke("Raise of AI in india")
print(result)