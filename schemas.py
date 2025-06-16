from pydantic import BaseModel
from typing import List


class RegionLore(BaseModel):
    region_name: str
    lore: str


class Location(BaseModel):
    name: str
    description: str


class Locations(BaseModel):
    locations: List[Location]


class Faction(BaseModel):
    name: str
    description: str
    goals: str
    conflicts: str


class Factions(BaseModel):
    factions: List[Faction]


class NPC(BaseModel):
    name: str
    role: str
    motivation: str


class NPCs(BaseModel):
    npcs: List[NPC]
