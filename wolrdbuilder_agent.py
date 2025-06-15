import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

load_dotenv()

def generate_region_lore(state:dict):
    llm = ChatOpenAI(model="gpt-4o")
    prompt = f"Create the lore for a fantasy region called '{state['region_name']}'. Include geography, climate, cultures, and themes."
    response = llm.invoke(prompt)
    state['lore'] = response.content
    return state

def generate_locations(state:dict):
    llm = ChatOpenAI(model="gpt-4o")
    prompt = f"Based on the following lore:\n{state['lore']}\n\nGenerate 5 major locations (cities, ruins, landmarks) for the region '{state['region_name']}'."
    response = llm.invoke(prompt)
    state['locations'] = response.content
    return state

def generate_factions(state:dict):
    llm = ChatOpenAI(model="gpt-4o")
    prompt = f"Based on the region '{state['region_name']}' and these locations:\n{state['locations']}\n\nGenerate 3 major factions or powers with goals and conflicts."
    response = llm.invoke(prompt)
    state['factions'] = response.content
    return state

def generate_npcs(state:dict):
    llm = ChatOpenAI(model="gpt-4o")
    prompt = f"Based on this region '{state['region_name']}' with its lore, locations, and factions:\nLore: {state['lore']}\nLocations: {state['locations']}\nFactions: {state['factions']}\n\nGenerate 5 major NPCs with name, role, and motivation."
    response = llm.invoke(prompt)
    state['npcs'] = response.content
    return state

# TODO: def save_world(state: WorldState):

graph = StateGraph(dict)

graph.add_node("lore", generate_region_lore)
graph.add_node("locations", generate_locations)
graph.add_node("factions", generate_factions)
graph.add_node("npcs", generate_npcs)

graph.set_entry_point("lore")
graph.add_edge("lore", "locations")
graph.add_edge("locations", "factions")
graph.add_edge("factions", "npcs")

agent = graph.compile()

if __name__ == "__main__":
    user_region = input("Enter region name to build: ")
    state = {"region_name": user_region}
    final_state = agent.invoke(state)

    print("Worldbuilding Complete!")
    print("Region Name:", final_state['region_name'])
    print("Lore:", final_state['lore'])
    print("Locations:", final_state['locations'])
    print("Factions:", final_state['factions'])
    print("NPCs:", final_state['npcs'])