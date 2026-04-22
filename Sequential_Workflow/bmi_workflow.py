from langgraph.graph import StateGraph, START, END
from typing import TypedDict


#define states
class BMIStates(TypedDict):
    weight_kg:  float
    height_m: float
    bmi: float
    category: str


def calculate_bmi(state: BMIStates) -> BMIStates:
    weight = state['weight_kg']
    height = state['height_m']

    bmi = weight/(height**2)
    state['bmi'] = round(bmi,2)

    return state


def level_bmi(state: BMIStates) -> BMIStates:
    bmi = state['bmi']
    if bmi < 18.5:
        state['category'] = 'underweight'
    elif 18.5 <= bmi < 25:
        state['category'] = 'Normal'
    elif 25 <= bmi < 30:
        state['category'] = 'overweight'
    else:
        state['category'] = 'obese'
    return state

# define graph
graph = StateGraph(BMIStates)


#add nodes to your graph
graph.add_node('calculate_bmi', calculate_bmi)
graph.add_node('level_bmi', level_bmi)


#add edges to your graph
graph.add_edge(START,'calculate_bmi')
graph.add_edge('calculate_bmi','level_bmi')
graph.add_edge('level_bmi', END)


#compile the graph
workflow = graph.compile()

#execute the graph
final_state = workflow.invoke({'height_m':2, 'weight_kg':100})
print(final_state)

