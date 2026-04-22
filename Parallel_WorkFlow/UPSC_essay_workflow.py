from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import Field, BaseModel
import os
import operator


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm = llm)



essay = """
Technology has become an essential part of modern life. From smartphones and computers to artificial intelligence and the internet, technology influences how people live, work, and communicate. Over the years, it has brought many positive changes to society, but it has also created some challenges.

One of the biggest advantages of technology is improved communication. In the past, people relied on letters or landline phones to stay in touch. Today, with the help of the internet, social media, and messaging apps, people can communicate instantly across the world. This has made it easier to maintain relationships, share ideas, and collaborate on a global scale.

Technology has also transformed education. Students now have access to online learning platforms, digital libraries, and virtual classrooms. This allows them to learn anytime and anywhere. Teachers can use videos, presentations, and interactive tools to make learning more interesting and effective. As a result, education has become more accessible and flexible.

In the field of healthcare, technology has saved many lives. Advanced medical equipment, improved diagnostic tools, and telemedicine services help doctors treat patients more efficiently. People can now consult doctors online, which is especially helpful in remote areas. Technology also helps in researching new medicines and treatments.

Another important impact is in the workplace. Machines and software have made tasks faster and more efficient. Automation and artificial intelligence are helping businesses increase productivity. However, this has also led to concerns about job loss, as some traditional jobs are being replaced by machines.

Despite its benefits, technology also has some negative effects. Excessive use of smartphones and social media can lead to addiction and reduced physical activity. It can also affect mental health, causing stress, anxiety, and lack of concentration. Privacy and security are other major concerns, as personal data can be misused.

In conclusion, technology has greatly improved human life by making communication, education, healthcare, and work more efficient. However, it also brings challenges that need careful management. Society must use technology wisely to ensure that its benefits are maximized while minimizing its negative effects.
"""

prompt = f"""
Evaluate the following essay and return output ONLY in JSON format:

{{
  "feedback": "Detailed Feedback for essay",
  "score": number (0-10)
}}

Essay:
{essay}
"""


class UpscState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str 
    individual_score: Annotated[list[int], operator.add]
    avg_score: float


def evaluate_language(state: UpscState):
    prompt = f"""
Evaluate the following essay and return output ONLY in JSON format:

{{
  "feedback": "Detailed Feedback for essay",
  "score": number (0-10)
}}

Essay:
{essay}
"""
    output = model.invoke(prompt)
    return {'language_feedback':[output.feedback], 'individual_score': [output.score]}



def evaluate_analysis(state: UpscState):
    prompt = f"""
Evaluate the depth of analysis of the following essay and return output ONLY in JSON format:

{{
  "feedback": "Detailed Feedback for essay",
  "score": number (0-10)
}}

Essay:
{essay}
"""
    output = model.invoke(prompt)
    return {'analysis_feedback':[output.feedback], 'individual_score': [output.score]}




def evaluate_thought(state: UpscState):
    prompt = f"""
Evaluate the clarity of thought of the following essay and return output ONLY in JSON format:

{{
  "feedback": "Detailed Feedback for essay",
  "score": number (0-10)
}}

Essay:
{essay}
"""
    output = model.invoke(prompt)
    return {'clarity_feedback':[output.feedback], 'individual_score': [output.score]}



def final_evaluation(state: UpscState):
    #summary of feedback
    prompt = f"based on the following feedback create a summarized feedback \n language feedback - {state['language_feedback']} \n depth of analysis feedback - {state['analysis_feedback']}  \n clarity of thought feedback - {state['clarity_feedback']}"

    overall_feedback = model.invoke(prompt).content
    
    #avg calculation
    avg_score = sum(state['individual_score'])/len(state['individual_score'])

    return {'overall_feedback': overall_feedback, 'avg_score': avg_score}

graph = StateGraph(UpscState)

graph.add_node('evaluate_language',evaluate_language)
graph.add_node('evaluate_analysis',evaluate_analysis)
graph.add_node('evaluate_thought',evaluate_thought)
graph.add_node('final_evaluation',final_evaluation)


graph.add_edge(START,'evaluate_language')
graph.add_edge(START,'evaluate_analysis')
graph.add_edge(START,'evaluate_thought')
graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_thought', 'final_evaluation')
graph.add_edge('final_evaluation',END)

workflow = graph.compile()
initial_state = {
    "essay": essay
}

final_state = workflow.invoke(initial_state)
print(final_state)



#result = model.invoke(prompt)
#print(result.content)

