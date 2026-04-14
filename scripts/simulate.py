#!/usr/bin/env python3
"""
Interactive call simulator — test the IVR from your terminal.

Simulates a phone call: you can type digit presses (1, 2, 3...)
or speak in natural language. The system responds just like it
would on a real call.

Usage:
    python scripts/simulate.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from danooc.config import Config
from danooc.ivr.engine import CallFlowEngine
from danooc.nlp.classifier import IntentClassifier


BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def main() -> None:
    config = Config()
    engine = CallFlowEngine.from_json(config.call_flow_path, max_retries=config.max_retries)

    clf = IntentClassifier()
    try:
        clf.load(config.model_path)
        has_nlp = True
    except FileNotFoundError:
        print(f"{YELLOW}[!] No se encontró modelo de intenciones. Entrena primero con: python scripts/train.py{RESET}")
        has_nlp = False

    call_sid = "sim-001"
    prompt, node_id = engine.start_call(call_sid)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  DANOOC — Simulador de Llamada{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BLUE}  Escribe un número (1-9, 0) para presionar teclas{RESET}")
    print(f"{BLUE}  Escribe texto para hablar con lenguaje natural{RESET}")
    print(f"{BLUE}  Escribe 'salir' para colgar{RESET}")
    print(f"{'='*60}\n")

    print(f"{GREEN}📞 Sistema: {prompt}{RESET}\n")

    while True:
        try:
            user_input = input(f"{BOLD}Tú: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{RED}📵 Llamada terminada.{RESET}")
            break

        if not user_input:
            result = engine.handle_no_input(call_sid)
        elif user_input.lower() in ("salir", "colgar", "exit", "quit"):
            print(f"\n{RED}📵 Llamada terminada. Gracias por llamar a Danooc.{RESET}")
            break
        elif user_input.isdigit() and len(user_input) == 1:
            result = engine.handle_dtmf(call_sid, user_input)
        else:
            intent = None
            if has_nlp:
                intent, conf = clf.predict(user_input)
                print(f"{YELLOW}  [NLP] intent={intent}, confianza={conf:.2%}{RESET}")
                if conf < config.confidence_threshold:
                    intent = None
            result = engine.handle_speech(call_sid, user_input, intent=intent)

        action = result.get("action", "gather")
        response_prompt = result.get("prompt", "")
        target = result.get("action_target", "")

        print(f"\n{GREEN}📞 Sistema: {response_prompt}{RESET}")

        if action == "transfer":
            print(f"{YELLOW}  [→ Transferir a: {target}]{RESET}")
            print(f"\n{RED}📵 Llamada transferida.{RESET}")
            break
        elif action == "hangup":
            print(f"\n{RED}📵 Llamada terminada.{RESET}")
            break
        elif action == "voicemail":
            print(f"{YELLOW}  [🎙️ Grabando mensaje de voz...]{RESET}")
            try:
                msg = input(f"{BOLD}Tu mensaje: {RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                pass
            print(f"{GREEN}📞 Sistema: Gracias por su mensaje. Hasta luego.{RESET}")
            print(f"\n{RED}📵 Llamada terminada.{RESET}")
            break

        print()

    engine.end_call(call_sid)


if __name__ == "__main__":
    main()
