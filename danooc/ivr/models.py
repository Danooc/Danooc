"""
Data models for IVR call flows.

A call flow is a tree of nodes. Each node is either:
  - MenuNode: plays a prompt and waits for DTMF digits or speech input
  - ActionNode: performs an action (transfer, hangup, voicemail, webhook)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class NodeType(str, Enum):
    MENU = "menu"
    ACTION = "action"


class ActionType(str, Enum):
    TRANSFER = "transfer"
    HANGUP = "hangup"
    VOICEMAIL = "voicemail"
    WEBHOOK = "webhook"
    REPEAT = "repeat"
    BACK = "back"


@dataclass
class MenuOption:
    """A single option within a menu (e.g. 'Press 1 for sales')."""
    digit: str
    label: str
    next_node: str
    intents: List[str] = field(default_factory=list)


@dataclass
class MenuNode:
    id: str
    prompt: str
    options: List[MenuOption] = field(default_factory=list)
    no_input_prompt: str = "No escuché su respuesta."
    invalid_prompt: str = "Opción no válida."
    max_retries: int = 3
    node_type: NodeType = NodeType.MENU

    def get_option_by_digit(self, digit: str) -> Optional[MenuOption]:
        for opt in self.options:
            if opt.digit == digit:
                return opt
        return None

    def get_option_by_intent(self, intent: str) -> Optional[MenuOption]:
        for opt in self.options:
            if intent in opt.intents:
                return opt
        return None


@dataclass
class ActionNode:
    id: str
    prompt: str
    action: ActionType
    action_target: str = ""
    node_type: NodeType = NodeType.ACTION


@dataclass
class CallFlow:
    """Complete call flow definition."""
    name: str
    welcome_message: str
    root_node: str
    nodes: Dict[str, MenuNode | ActionNode] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> CallFlow:
        nodes: Dict[str, MenuNode | ActionNode] = {}
        for ndata in data.get("nodes", []):
            ntype = ndata.get("node_type", "menu")
            if ntype == "action":
                nodes[ndata["id"]] = ActionNode(
                    id=ndata["id"],
                    prompt=ndata.get("prompt", ""),
                    action=ActionType(ndata.get("action", "hangup")),
                    action_target=ndata.get("action_target", ""),
                    node_type=NodeType.ACTION,
                )
            else:
                options = [
                    MenuOption(
                        digit=o["digit"],
                        label=o["label"],
                        next_node=o["next_node"],
                        intents=o.get("intents", []),
                    )
                    for o in ndata.get("options", [])
                ]
                nodes[ndata["id"]] = MenuNode(
                    id=ndata["id"],
                    prompt=ndata.get("prompt", ""),
                    options=options,
                    no_input_prompt=ndata.get("no_input_prompt", "No escuché su respuesta."),
                    invalid_prompt=ndata.get("invalid_prompt", "Opción no válida."),
                    max_retries=ndata.get("max_retries", 3),
                    node_type=NodeType.MENU,
                )
        return cls(
            name=data.get("name", "default"),
            welcome_message=data.get("welcome_message", ""),
            root_node=data.get("root_node", ""),
            nodes=nodes,
        )
