import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from world_storage import WorldStorage
from world_logic import  agent
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel
from typing import Dict

load_dotenv()
llm = ChatOpenAI(model="gpt-4o")

VALID_INTENTS = {
    "create_world": "create a new world or region",
    "load_world": "load or switch to a saved world",
    "list_worlds": "show all available saved worlds",
    "describe_current_world": "describe the loaded world",
    "inquire_command_help": "ask about command usage or what commands exist",
    "small_talk": "casual conversation",
    "unknown": "fallback for unclear input"
}

VALID_SLOTS = [
    "world_name",
    "region_name",
    "command_name"
]

class IntentOutput(BaseModel):
    intent: str
    slots: Dict[str, str]

intent_parser = OutputFixingParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=IntentOutput),
    llm=llm
)

current_state = {}

chat_history = [
    {"role": "system", "content": (
        "You are a D&D worldbuilding assistant. You help users manage their fantasy worlds, including regions, NPCs, lore, factions, and more. "
        "You are not a general-purpose assistant. You live inside a D&D campaign builder. Your job is to support fantasy world creation, loading, listing, and describing regions using natural language."
    )}
]

command_help_doc = {
    "create_world": "Creates a new region in your world. You can say: 'Create a new world named Mythra'.",
    "load_world": "Loads a previously saved world into memory. You can say: 'Load world Svenlia'.",
    "list_worlds": "Displays all saved world names that you can load.",
    "describe_current_world": "Shows all details (lore, locations, NPCs, etc.) of the currently loaded world.",
}

def interpret_user_input(user_input: str) -> dict:
    format_instructions = intent_parser.parser.get_format_instructions()

    valid_intents_text = "\n".join(f"- {intent}: {desc}" for intent,desc in VALID_INTENTS.items())
    valid_slots_text = "\n".join(f"* {slot}" for slot in VALID_SLOTS)

    prompt = (
        "You are a D&D worldbuilding assistant. Classify the user's message into a command intent and extract any slot values.\n\n"
        f"Valid intents:\n{valid_intents_text}\n\n"
        f"Valid slot names:\n{valid_slots_text}\n\n"
        "If the user says something like 'What does list worlds do?', your intent should be 'inquire_command_help' and the slot should be command_name='list_worlds'.\n\n"
        f"User input: {user_input}\n\n"
        f"Respond with valid JSON in this format:\n{format_instructions}"
    )

    try:
        result = intent_parser.invoke(prompt)
        return result.model_dump()
    except Exception as e:
        print("[DEBUG] Parser error:", e)
        return {"intent": "unknown", "slots": {}}


def handle_command(parsed: dict, user_input: str):
    global current_state
    intent = parsed.get("intent", "unknown")
    slots = parsed.get("slots", {})

    if intent == "create_world":
        name = slots.get("region_name")
        if not name:
            print("What should the region be called?")
            return
        state = {"region_name": name}
        current_state = agent.invoke(state)
        print(f"World '{name}' created.")

    elif intent == "list_worlds":
        worlds = WorldStorage.list_worlds()
        print("Available worlds:", worlds)

    elif intent == "load_world":
        known_worlds = WorldStorage.list_worlds()
        raw_input_name = slots.get("world_name") or ""

        matched = next((w for w in known_worlds if w.lower() in raw_input_name.lower()), None)

        if not matched:
            print(f"I couldn't find the world you're referring to. Did you mean one of these? {known_worlds}")
            return

        try:
            current_state = WorldStorage.load_world(matched)
            print(f"World '{matched}' loaded.")
        except FileNotFoundError:
            print(f"World '{matched}' not found.")

    elif intent == "describe_current_world":
        if not current_state:
            print("No world currently loaded.")
        else:
            print(json.dumps(current_state, indent=4, ensure_ascii=False))

    elif intent == "inquire_command_help":
        cmd = slots.get("command_name", "").lower()
        if cmd and cmd in command_help_doc:
            print(command_help_doc[cmd])
        else:
            print("Here are the commands I support:")
            for name, desc in command_help_doc.items():
                print(f"- {name}: {desc}")

    elif intent == "small_talk":
        chat_history.append({"role": "user", "content": user_input})
        response = llm.invoke(chat_history)
        chat_history.append({"role": "assistant", "content": response.content})
        print(response.content)

    elif intent == "unknown":
        chat_history.append({"role": "user", "content": user_input})
        response = llm.invoke(chat_history)
        chat_history.append({"role": "assistant", "content": response.content})
        print(response.content)


def chat_loop():
    while True:
        user_input = input("\n> ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        parsed = interpret_user_input(user_input)
        print("[DEBUG] Parsed Intent:", parsed)
        handle_command(parsed, user_input)


if __name__ == "__main__":
    chat_loop()