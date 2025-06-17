import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from world_manager import WorldPersistence
from wolrdbuilder_agent import  agent

load_dotenv()
llm = ChatOpenAI(model="gpt-4o")

current_state = {}

def interpret_command(user_input: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that classifies worldbuilding-related user input."),
        ("user", "User input: {input}\nClassify it into one of these commands:\n"
                 "- create_world\n"
                 "- list_worlds\n"
                 "- load_world\n"
                 "- describe_current_world\n"
                 "- unknown\nReturn ONLY the command string."),
    ])
    formatted = prompt.format_messages(input=user_input)
    response = llm.invoke(formatted)
    return response.content.strip()

def handle_command(command: str, user_input: str):
    global current_state

    if command == "create_world":
        name = input("Enter region name: ")
        state = {"region_name": name}
        current_state = agent.invoke(state)
        print(f"World '{name}' created.")

    elif command == "list_worlds":
        worlds = WorldPersistence.list_worlds()
        print("Available worlds:", worlds)

    elif command == "load_world":
        name = input("Enter world name to load: ")
        current_state = WorldPersistence.load_world(name)
        print(f"World '{name}' loaded.")

    elif command == "describe_current_world":
        if not current_state:
            print("No world currently loaded.")
        else:
            print(json.dumps(current_state, indent=4, ensure_ascii=False))

    elif command == "unknown":
        print("Sorry, I didn't understand that command.")

def chat_loop():
    print("Welcome to the D&D Worldbuilder Chat Interface!")
    print("I am a world building AI which you can use to do things like:\n- create a new world\n- load world X\n- show my current world")

    while True:
        user_input = input("\n> ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        command = interpret_command(user_input)
        handle_command(command, user_input)

if __name__ == "__main__":
    chat_loop()