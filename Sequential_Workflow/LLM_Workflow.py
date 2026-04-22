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


class LLMState(TypedDict):
    questions : str
    answer : str


def llm_qa(state: LLMState) -> LLMState:
    #extract the question from state
    questions = state["questions"]

    # form a prompt
     
    prompt = f"answer the following questions {questions}"
    
    # ask the questions to the LLM

    answer = model.invoke(prompt)

    # update answer in the state
    state["answer"] = answer

    return state



graph = StateGraph(LLMState)

graph.add_node('llm_qa', llm_qa)


graph.add_edge(START, 'llm_qa')
graph.add_edge('llm_qa',END)

#workflow = graph.compile()

#final_state = workflow.invoke({'questions':'How far the moon from the earth'})

#print(final_state['answer'])

result = model.invoke('How far the moon from the earth')
print(result)