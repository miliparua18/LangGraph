from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from dotenv import load_dotenv
from model import SubGraphmodel

load_dotenv()

submodel = SubGraphmodel()


class Substate(TypedDict):
    input_text: str
    translated_text: str


def translate_text(state: Substate):

    prompt = f"""
    translate the following text in hindi.
    Keep it natural and clear.
    Do not add extra content.

    Text:
    {state["input_text"]}
    """.strip()

    translated_text = submodel.invoke(prompt).content

    return {
        'translated_text': translated_text
    }


Subgraph_builder = StateGraph(Substate)

Subgraph_builder.add_node("translate_text", translate_text)

Subgraph_builder.add_edge(START, "translate_text")
Subgraph_builder.add_edge("translate_text", END)

Subgraph = Subgraph_builder.compile()


class ParentState(TypedDict):
    question: str
    answer_eng: str
    answer_hin: str


parentmodel = SubGraphmodel()


def Generate(state: ParentState):

    answer = parentmodel.invoke(
        f"You are a helpful assistant. Answer clearly.\n\nQuestion: {state['question']}"
    ).content

    return {
        'answer_eng': answer
    }


def Translate(state: ParentState):

    result = Subgraph.invoke({
        'input_text': state['answer_eng']
    })

    return {
        'answer_hin': result['translated_text']
    }


parentgraph_builder = StateGraph(ParentState)

parentgraph_builder.add_node("Generate", Generate)
parentgraph_builder.add_node("Translate", Translate)

parentgraph_builder.add_edge(START, "Generate")
parentgraph_builder.add_edge("Generate", "Translate")
parentgraph_builder.add_edge("Translate", END)

parentgraph = parentgraph_builder.compile()

final_result = parentgraph.invoke({
    'question': 'what is quantum physics'
})

print(final_result)