import os
import json
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

def parse_json_response(response):
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        print("Warning: LLM output not valid JSON! trying to extract JSON from text...")

        try:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                print("No JSON object found.")
                return response.content
        except Exception as e:
            print(f"Failed to parse JSON fallback: {e}")
            return response.content

def generate_region_lore(state: dict):
    prompt = f"""
    You are a worldbuilding AI. Generate detailed LORE for a fantasy region called "{state['region_name']}". 
    Return ONLY THE valid json in the following format:
    {{
    "region_name": "{state['region_name']}",
    "lore": "...full lore text here..."
    }}
    """
    response = llm.invoke(prompt)
    data = parse_json_response(response)
    state['lore'] = data['lore']
    return state

def generate_locations(state: dict):
    prompt = f"""
    You are a worldbuilding AI. Generate 5 major locations for the region "{state['region_name']}" based on this lore:

    LORE: {state['lore']}

    Return ONLY THE valid json as:
    {{
    "locations": [
        {{"name": "...", "description": "..."}},
        {{"name": "...", "description": "..."}},
        {{"name": "...", "description": "..."}},
        {{"name": "...", "description": "..."}},
        {{"name": "...", "description": "..."}}
    ]
    }}
    """
    response = llm.invoke(prompt)
    data = parse_json_response(response)
    state['locations'] = data['locations']
    return state

def generate_factions(state: dict):
    prompt = f"""
    You are a worldbuilding AI. Generate 3 major factions for the region "{state['region_name']}" based on its lore and locations.

    LORE: {state['lore']}
    LOCATIONS: {state['locations']}

    Return ONLY THE valid json as:
    {{
    "factions": [
        {{
        "name": "...",
        "description": "...",
        "goals": "...",
        "conflicts": "..."
        }},
        {{
        "name": "...",
        "description": "...",
        "goals": "...",
        "conflicts": "..."
        }},
        {{
        "name": "...",
        "description": "...",
        "goals": "...",
        "conflicts": "..."
        }}
    ]
    }}
    """
    response = llm.invoke(prompt)
    data = parse_json_response(response)
    state['factions'] = data['factions']
    return state

def generate_npcs(state: dict):
    prompt = f"""
    You are a worldbuilding AI. Generate 5 major NPCs for the region "{state['region_name']}" based on its lore, locations, and factions.

    LORE: {state['lore']}
    LOCATIONS: {state['locations']}
    FACTIONS: {state['factions']}

    Return ONLY THE valid json as:
    {{
    "npcs": [
        {{"name": "...", "role": "...", "motivation": "..."}},
        {{"name": "...", "role": "...", "motivation": "..."}},
        {{"name": "...", "role": "...", "motivation": "..."}},
        {{"name": "...", "role": "...", "motivation": "..."}},
        {{"name": "...", "role": "...", "motivation": "..."}}
    ]
    }}
    """
    response = llm.invoke(prompt)
    data = parse_json_response(response)
    state['npcs'] = data['npcs']
    return state

def save_world(state: dict):
    world_data = {
        "region": state['region_name'],
        "lore": state['lore'],
        "locations": state['locations'],
        "factions": state['factions'],
        "npcs": state['npcs']
    }

    filename = f"{state['region_name'].replace(' ', '_').lower()}_world.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(world_data, f, indent=4, ensure_ascii=False)

    print(f"World saved to: {filename}")
    return state


graph = StateGraph(dict)

graph.add_node("lore", generate_region_lore)
graph.add_node("locations", generate_locations)
graph.add_node("factions", generate_factions)
graph.add_node("npcs", generate_npcs)
graph.add_node("save", save_world)

graph.set_entry_point("lore")
graph.add_edge("lore", "locations")
graph.add_edge("locations", "factions")
graph.add_edge("factions", "npcs")
graph.add_edge("npcs", "save")

agent = graph.compile()

if __name__ == "__main__":
    user_region = input("Enter region name to build: ")
    state = {"region_name": user_region}
    final_state = agent.invoke(state)

    print("Worldbuilding Complete!")
    print(json.dumps(final_state, indent=4, ensure_ascii=False))