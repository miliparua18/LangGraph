from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict, Literal
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

#State
class ReviewState(TypedDict):
    review: str
    sentiment: Literal['positive', 'negative', 'neutral']
    diagnosis: str
    response: str

#Node 1: Sentiment
def find_sentiment(state: ReviewState):
    prompt = f"""
Classify sentiment (positive, negative, neutral) for:
{state['review']}
"""
    result = model.invoke(prompt)
    sentiment = result.content.lower()

    if "positive" in sentiment:
        sentiment = "positive"
    elif "negative" in sentiment:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {'sentiment': sentiment}

#Node 2: Positive response
def positive_response(state: ReviewState):
    prompt = f"Write a warm thank you message:\n{state['review']}"
    result = model.invoke(prompt)
    return {'response': result.content}

#Node 3: Negative response
def negative_response(state: ReviewState):
    prompt = f"Write an apology message:\n{state['review']}"
    result = model.invoke(prompt)
    return {'response': result.content}

#Node 4: Diagnosis
def run_diagnosis(state: ReviewState):
    return {'diagnosis': f"User sentiment is {state['sentiment']}"}

#Condition function
def route(state: ReviewState):
    if state['sentiment'] == "positive":
        return "positive_response"
    elif state['sentiment'] == "negative":
        return "run_diagnosis"   # 👈 goes to diagnosis first
    else:
        return END

#Graph
graph = StateGraph(ReviewState)

graph.add_node('find_sentiment', find_sentiment)
graph.add_node('positive_response', positive_response)
graph.add_node('negative_response', negative_response)
graph.add_node('run_diagnosis', run_diagnosis)

graph.add_edge(START, 'find_sentiment')

#Conditional routing
graph.add_conditional_edges('find_sentiment', route)

#Chain negative flow
graph.add_edge('run_diagnosis', 'negative_response')

#End connections
graph.add_edge('positive_response', END)
graph.add_edge('negative_response', END)

#Compile
workflow = graph.compile()

#Input
initial_state = {
    'review': 'The product was very bad'
}

#Run
result = workflow.invoke(initial_state)

print(result)