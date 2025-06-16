import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from schemas import RegionLore, Locations, Factions, NPCs

load_dotenv()
llm = ChatOpenAI(model="gpt-4o")

def build_output_fixing_chain(llm, schema, prompt_template_str, input_variables):
    pydantic_parser = PydanticOutputParser(pydantic_object=schema)

    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=input_variables,
        partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
    )

    fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

    chain = prompt | llm | fixing_parser
    return chain


def generate_region_lore(state):
    lore_prompt_str = """
You are a worldbuilding AI. Generate detailed LORE for a fantasy region called "{region_name}". 
    
Return ONLY valid JSON:
{format_instructions}
    """
    chain = build_output_fixing_chain(
        llm, RegionLore, lore_prompt_str, input_variables=["region_name"]
    )
    output = chain.invoke({"region_name": state['region_name']})
    state['lore'] = output.lore
    return state


def generate_locations(state):
    locations_prompt_str = """
You are a worldbuilding AI. Generate 5 major locations for the region "{region_name}" based on this lore:

LORE: {lore}

Return ONLY valid JSON:
{format_instructions}
"""
    chain = build_output_fixing_chain(
        llm, Locations, locations_prompt_str, input_variables=["region_name", "lore"]
    )
    output = chain.invoke({
        "region_name": state['region_name'],
        "lore": state['lore']
    })
    state['locations'] = [loc.model_dump() for loc in output.locations]
    return state


def generate_factions(state):
    factions_prompt_str = """
You are a worldbuilding AI. Generate 3 major factions for the region "{region_name}" based on its lore and locations.

LORE: {lore}
LOCATIONS: {locations}

Return ONLY valid JSON:
{format_instructions}
"""
    chain = build_output_fixing_chain(
        llm, Factions, factions_prompt_str, input_variables=["region_name", "lore", "locations"]
    )
    output = chain.invoke({
        "region_name": state['region_name'],
        "lore": state['lore'],
        "locations": json.dumps(state['locations'])
    })
    state['factions'] = [f.model_dump() for f in output.factions]
    return state


def generate_npcs(state):
    npcs_prompt_str = """
You are a worldbuilding AI. Generate 5 major NPCs for the region "{region_name}" based on its lore, locations, and factions.

LORE: {lore}
LOCATIONS: {locations}
FACTIONS: {factions}

Return ONLY valid JSON:
{format_instructions}
"""
    chain = build_output_fixing_chain(
        llm, NPCs, npcs_prompt_str, input_variables=["region_name", "lore", "locations", "factions"]
    )
    output = chain.invoke({
        "region_name": state['region_name'],
        "lore": state['lore'],
        "locations": json.dumps(state['locations']),
        "factions": json.dumps(state['factions'])
    })
    state['npcs'] = [npc.model_dump() for npc in output.npcs]
    return state


def save_world(state):
    world_data = {
        "region": state['region_name'],
        "lore": state['lore'],
        "locations": state['locations'],
        "factions": state['factions'],
        "npcs": state['npcs']
    }

    os.makedirs("worlds", exist_ok=True)
    filename = f"worlds/{state['region_name'].replace(' ', '_').lower()}_world.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(world_data, f, indent=4, ensure_ascii=False)

    print(f"World saved to: {filename}")
    return state


# BUILD THE LANGGRAPH PIPELINE
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
