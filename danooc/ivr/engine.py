"""
Call flow engine — drives a call through the IVR state machine.

Each active call is tracked by ``call_sid``. The engine resolves user input
(DTMF digits or free-text classified by the intent model) to navigate the
tree of MenuNode / ActionNode.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from danooc.ivr.models import (
    ActionNode,
    ActionType,
    CallFlow,
    MenuNode,
    NodeType,
)

logger = logging.getLogger(__name__)


@dataclass
class CallState:
    call_sid: str
    current_node: str
    retries: int = 0
    history: list = field(default_factory=list)
    context: dict = field(default_factory=dict)


class CallFlowEngine:
    """
    Stateful engine that manages active calls through an IVR flow.
    """

    def __init__(self, flow: CallFlow, max_retries: int = 3) -> None:
        self.flow = flow
        self.max_retries = max_retries
        self._calls: Dict[str, CallState] = {}

    # ------------------------------------------------------------------
    # Flow loading
    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str, max_retries: int = 3) -> CallFlowEngine:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        flow = CallFlow.from_dict(data)
        return cls(flow, max_retries)

    # ------------------------------------------------------------------
    # Call lifecycle
    # ------------------------------------------------------------------
    def start_call(self, call_sid: str) -> Tuple[str, str]:
        """
        Start a new call. Returns ``(prompt_text, current_node_id)``.
        """
        state = CallState(call_sid=call_sid, current_node=self.flow.root_node)
        self._calls[call_sid] = state

        node = self.flow.nodes[self.flow.root_node]
        greeting = self.flow.welcome_message
        prompt = self._build_menu_prompt(node) if isinstance(node, MenuNode) else node.prompt
        full_prompt = f"{greeting} {prompt}".strip()
        state.history.append(("system", full_prompt))
        return full_prompt, state.current_node

    def handle_dtmf(self, call_sid: str, digit: str) -> dict:
        """Process a DTMF digit press."""
        return self._handle_input(call_sid, digit=digit)

    def handle_speech(self, call_sid: str, text: str, intent: Optional[str] = None) -> dict:
        """Process speech input (transcribed text and/or classified intent)."""
        return self._handle_input(call_sid, speech=text, intent=intent)

    def handle_no_input(self, call_sid: str) -> dict:
        """Handle timeout / no input."""
        state = self._get_state(call_sid)
        node = self.flow.nodes[state.current_node]

        state.retries += 1
        if state.retries >= self.max_retries:
            return self._result(
                state,
                prompt="Lo sentimos, no pudimos recibir su respuesta. La llamada será transferida a un agente.",
                action="transfer",
                action_target="operator",
            )

        no_input_msg = node.no_input_prompt if isinstance(node, MenuNode) else "No escuché su respuesta."
        menu_prompt = self._build_menu_prompt(node) if isinstance(node, MenuNode) else ""
        prompt = f"{no_input_msg} {menu_prompt}".strip()
        state.history.append(("system", prompt))
        return self._result(state, prompt=prompt, action="gather")

    def end_call(self, call_sid: str) -> None:
        self._calls.pop(call_sid, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _get_state(self, call_sid: str) -> CallState:
        state = self._calls.get(call_sid)
        if not state:
            raise KeyError(f"No active call with sid={call_sid}")
        return state

    def _handle_input(
        self,
        call_sid: str,
        digit: Optional[str] = None,
        speech: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> dict:
        state = self._get_state(call_sid)
        node = self.flow.nodes[state.current_node]

        if not isinstance(node, MenuNode):
            return self._result(state, prompt=node.prompt, action=node.action.value, action_target=node.action_target)

        option = None
        if digit:
            state.history.append(("user_dtmf", digit))
            option = node.get_option_by_digit(digit)
        if not option and intent:
            state.history.append(("user_speech", speech or ""))
            option = node.get_option_by_intent(intent)
        if not option and speech and not intent:
            state.history.append(("user_speech", speech))

        if not option:
            state.retries += 1
            if state.retries >= self.max_retries:
                return self._result(
                    state,
                    prompt="Lo sentimos, no pudimos entender su respuesta. Será transferido a un agente.",
                    action="transfer",
                    action_target="operator",
                )
            invalid_msg = node.invalid_prompt
            menu_prompt = self._build_menu_prompt(node)
            prompt = f"{invalid_msg} {menu_prompt}".strip()
            state.history.append(("system", prompt))
            return self._result(state, prompt=prompt, action="gather")

        state.retries = 0
        next_node = self.flow.nodes.get(option.next_node)
        if not next_node:
            logger.error("Node %s not found in flow", option.next_node)
            return self._result(state, prompt="Error interno. Transferido a un agente.", action="transfer", action_target="operator")

        state.current_node = next_node.id

        if isinstance(next_node, ActionNode):
            state.history.append(("system", next_node.prompt))
            return self._result(state, prompt=next_node.prompt, action=next_node.action.value, action_target=next_node.action_target)

        prompt = self._build_menu_prompt(next_node)
        state.history.append(("system", prompt))
        return self._result(state, prompt=prompt, action="gather")

    @staticmethod
    def _build_menu_prompt(node: MenuNode) -> str:
        lines = [node.prompt] if node.prompt else []
        for opt in node.options:
            lines.append(f"Presione {opt.digit} para {opt.label}.")
        return " ".join(lines)

    @staticmethod
    def _result(state: CallState, *, prompt: str, action: str = "gather", action_target: str = "") -> dict:
        return {
            "call_sid": state.call_sid,
            "node": state.current_node,
            "prompt": prompt,
            "action": action,
            "action_target": action_target,
        }
