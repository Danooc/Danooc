"""
TwiML response builder — generates Twilio-compatible XML responses.

Supports: Say (TTS), Gather (DTMF + speech), Dial (transfer), Hangup.
"""
from __future__ import annotations

from xml.etree.ElementTree import Element, SubElement, tostring


class TwiMLBuilder:
    """Build TwiML XML responses for Twilio."""

    def __init__(
        self,
        voice: str = "Polly.Mia",
        language: str = "es-MX",
        base_url: str = "http://localhost:8000",
        input_timeout: int = 5,
        speech_timeout: str = "auto",
    ) -> None:
        self.voice = voice
        self.language = language
        self.base_url = base_url
        self.input_timeout = input_timeout
        self.speech_timeout = speech_timeout

    def gather_response(self, prompt: str, action_path: str = "/voice/handle-input") -> str:
        root = Element("Response")
        gather = SubElement(root, "Gather", {
            "input": "dtmf speech",
            "timeout": str(self.input_timeout),
            "speechTimeout": self.speech_timeout,
            "numDigits": "1",
            "language": self.language,
            "action": f"{self.base_url}{action_path}",
            "method": "POST",
        })
        say = SubElement(gather, "Say", {"voice": self.voice, "language": self.language})
        say.text = prompt

        redirect = SubElement(root, "Redirect", {"method": "POST"})
        redirect.text = f"{self.base_url}/voice/no-input"

        return self._to_xml(root)

    def say_and_hangup(self, prompt: str) -> str:
        root = Element("Response")
        say = SubElement(root, "Say", {"voice": self.voice, "language": self.language})
        say.text = prompt
        SubElement(root, "Hangup")
        return self._to_xml(root)

    def transfer(self, prompt: str, number: str) -> str:
        root = Element("Response")
        if prompt:
            say = SubElement(root, "Say", {"voice": self.voice, "language": self.language})
            say.text = prompt
        dial = SubElement(root, "Dial")
        num = SubElement(dial, "Number")
        num.text = number
        return self._to_xml(root)

    def voicemail(self, prompt: str, action_path: str = "/voice/voicemail") -> str:
        root = Element("Response")
        say = SubElement(root, "Say", {"voice": self.voice, "language": self.language})
        say.text = prompt
        SubElement(root, "Record", {
            "maxLength": "120",
            "action": f"{self.base_url}{action_path}",
            "method": "POST",
            "transcribe": "true",
        })
        return self._to_xml(root)

    def say_response(self, prompt: str) -> str:
        root = Element("Response")
        say = SubElement(root, "Say", {"voice": self.voice, "language": self.language})
        say.text = prompt
        return self._to_xml(root)

    @staticmethod
    def _to_xml(root: Element) -> str:
        return '<?xml version="1.0" encoding="UTF-8"?>' + tostring(root, encoding="unicode")
