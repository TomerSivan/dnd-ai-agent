import os
import json
from typing import List
import shutil

class WorldStorage:

    WORLDS_FOLDER = "worlds"

    @staticmethod
    def save_world(state:dict):
        world_name = state['region_name']
        world_path = os.path.join(WorldStorage.WORLDS_FOLDER, world_name)
        os.makedirs(world_path, exist_ok=True)

        with open(os.path.join(world_path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"region_name": world_name}, f, indent=4)

        with open(os.path.join(world_path, "lore.json"), "w", encoding="utf-8") as f:
            json.dump({"lore": state['lore']}, f, indent=4, ensure_ascii=False)

        with open(os.path.join(world_path, "locations.json"), "w", encoding="utf-8") as f:
            json.dump(state['locations'], f, indent=4, ensure_ascii=False)

        with open(os.path.join(world_path, "factions.json"), "w", encoding="utf-8") as f:
            json.dump(state['factions'], f, indent=4, ensure_ascii=False)

        with open(os.path.join(world_path, "npcs.json"), "w", encoding="utf-8") as f:
            json.dump(state['npcs'], f, indent=4, ensure_ascii=False)

        print(f"World '{world_name}' saved successfully!")

    @staticmethod
    def load_world(world_name:str) -> dict:
        world_path = os.path.join(WorldStorage.WORLDS_FOLDER, world_name)
        if not os.path.exists(world_path):
            raise FileNotFoundError(f"World '{world_name}' not found.")

        with open(os.path.join(world_path, "lore.json"), "r", encoding="utf-8") as f:
            lore = json.load(f)['lore']

        with open(os.path.join(world_path, "locations.json"), "r", encoding="utf-8") as f:
            locations = json.load(f)

        with open(os.path.join(world_path, "factions.json"), "r", encoding="utf-8") as f:
            factions = json.load(f)

        with open(os.path.join(world_path, "npcs.json"), "r", encoding="utf-8") as f:
            npcs = json.load(f)

        return {
            "region_name": world_name,
            "lore": lore,
            "locations": locations,
            "factions": factions,
            "npcs": npcs
        }

    @staticmethod
    def list_worlds() -> List[str]:
        if not os.path.exists(WorldStorage.WORLDS_FOLDER):
            return []
        return [
            name for name in os.listdir(WorldStorage.WORLDS_FOLDER)
            if os.path.isdir(os.path.join(WorldStorage.WORLDS_FOLDER, name))
        ]

    def delete_world_by_name(world_name:str):
        world_path = os.path.join(WorldStorage.WORLDS_FOLDER, world_name)
        if os.path.exists(world_path):
            shutil.rmtree(world_path)
            print(f"World '{world_name}' deleted.")
        else:
            print(f"World '{world_name}' does not exist.")
