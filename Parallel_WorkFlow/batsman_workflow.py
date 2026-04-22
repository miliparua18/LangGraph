from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class BatsManState(TypedDict):
    runs: int
    balls: int
    Fours: int
    Sixes: int
    sr: float
    bpb: float
    boundary_percent : float


def calculate_sr(state: BatsManState):
    sr = (state['runs']/state['balls'])*100
    state['sr'] = sr
    return {'sr': round(sr, 2)}


def calculate_bpb(state: BatsManState):
    bpb = state['balls']/(state['Fours']+state['Sixes'])
    state['bpb'] = bpb
    return {'bpb': round(bpb, 2)}

def calculate_boundary_percent(state: BatsManState):
    boundary_percent = (((state['Fours'] * 4) + (state['Sixes'] * 6))/state['runs'])*100
    state['boundary_percent'] = boundary_percent
    return {'boundary_percent': round(boundary_percent, 2)}


def summary(state: BatsManState):
    summary = f"""
strike_rate - {state['sr']} \n 
Ball per boundary - {state['bpb']} \n 
Boundary Percent - {state['boundary_percent']}
"""
    state['summary'] = summary
    return {'summary': summary}





graph = StateGraph(BatsManState)

graph.add_node('calculate_sr',calculate_sr)
graph.add_node('calculate_bpb',calculate_bpb)
graph.add_node('calculate_boundary_percent',calculate_boundary_percent)
graph.add_node('summary',summary)

graph.add_edge(START, 'calculate_sr')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_boundary_percent')
graph.add_edge('calculate_sr', 'summary')
graph.add_edge('calculate_bpb', 'summary')
graph.add_edge('calculate_boundary_percent', 'summary')
graph.add_edge('summary', END)


workflow = graph.compile()

final_state = workflow.invoke({'runs': 70, 'balls':6, 'Fours':3, 'Sixes': 5})
print(final_state)
