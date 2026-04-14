"""
FastAPI application — Twilio webhook endpoints for the IVR.

Endpoints:
  POST /voice/incoming   — Twilio calls this when a call arrives
  POST /voice/handle-input — receives DTMF or speech after a Gather
  POST /voice/no-input   — called on Gather timeout (no input)
  POST /voice/voicemail  — receives voicemail recording
  GET  /health           — health check
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import PlainTextResponse, Response

from danooc.config import Config
from danooc.ivr.engine import CallFlowEngine
from danooc.nlp.classifier import IntentClassifier
from danooc.tts.responses import TwiMLBuilder

logger = logging.getLogger(__name__)

_engine: Optional[CallFlowEngine] = None
_classifier: Optional[IntentClassifier] = None
_twiml: Optional[TwiMLBuilder] = None
_config: Optional[Config] = None


def create_app(config: Optional[Config] = None) -> FastAPI:
    global _engine, _classifier, _twiml, _config
    _config = config or Config()

    app = FastAPI(title="Danooc IVR", version="0.1.0")

    @app.on_event("startup")
    async def _startup() -> None:
        global _engine, _classifier, _twiml
        _engine = CallFlowEngine.from_json(_config.call_flow_path, max_retries=_config.max_retries)
        logger.info("Call flow loaded: %s", _engine.flow.name)

        _classifier = IntentClassifier()
        try:
            _classifier.load(_config.model_path)
            logger.info("Intent model loaded")
        except FileNotFoundError:
            logger.warning("No intent model found at %s — speech recognition will be disabled", _config.model_path)
            _classifier = None

        _twiml = TwiMLBuilder(
            voice=_config.tts_voice,
            language=_config.tts_language,
            base_url=_config.base_url,
            input_timeout=_config.input_timeout,
            speech_timeout=_config.speech_timeout,
        )

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------
    @app.post("/voice/incoming")
    async def incoming_call(CallSid: str = Form(default="")):
        call_sid = CallSid or str(uuid.uuid4())
        prompt, node_id = _engine.start_call(call_sid)
        logger.info("[%s] Call started → node=%s", call_sid, node_id)
        xml = _twiml.gather_response(prompt)
        return Response(content=xml, media_type="application/xml")

    @app.post("/voice/handle-input")
    async def handle_input(
        CallSid: str = Form(default=""),
        Digits: str = Form(default=""),
        SpeechResult: str = Form(default=""),
    ):
        call_sid = CallSid or "unknown"

        if call_sid not in _engine._calls:
            prompt, _ = _engine.start_call(call_sid)
            logger.info("[%s] Auto-started call (no prior incoming)", call_sid)

        if Digits:
            logger.info("[%s] DTMF: %s", call_sid, Digits)
            result = _engine.handle_dtmf(call_sid, Digits)
        elif SpeechResult:
            logger.info("[%s] Speech: %s", call_sid, SpeechResult)
            intent, confidence = None, 0.0
            if _classifier:
                intent, confidence = _classifier.predict(SpeechResult)
                logger.info("[%s] Intent: %s (%.2f)", call_sid, intent, confidence)
                if confidence < _config.confidence_threshold:
                    intent = None
            result = _engine.handle_speech(call_sid, SpeechResult, intent=intent)
        else:
            result = _engine.handle_no_input(call_sid)

        return _build_twiml_response(result)

    @app.post("/voice/no-input")
    async def no_input(CallSid: str = Form(default="")):
        call_sid = CallSid or "unknown"
        logger.info("[%s] No input received", call_sid)
        if call_sid not in _engine._calls:
            prompt, _ = _engine.start_call(call_sid)
            xml = _twiml.gather_response(prompt)
            return Response(content=xml, media_type="application/xml")
        result = _engine.handle_no_input(call_sid)
        return _build_twiml_response(result)

    @app.post("/voice/voicemail")
    async def voicemail(
        CallSid: str = Form(default=""),
        RecordingUrl: str = Form(default=""),
        RecordingDuration: str = Form(default="0"),
    ):
        logger.info("[%s] Voicemail received: %s (%ss)", CallSid, RecordingUrl, RecordingDuration)
        xml = _twiml.say_and_hangup("Gracias por su mensaje. Un agente se comunicará con usted pronto. Hasta luego.")
        return Response(content=xml, media_type="application/xml")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "flow": _engine.flow.name if _engine else None,
            "classifier": _classifier is not None,
        }

    return app


def _build_twiml_response(result: dict) -> Response:
    action = result.get("action", "gather")
    prompt = result.get("prompt", "")
    target = result.get("action_target", "")

    if action == "gather":
        xml = _twiml.gather_response(prompt)
    elif action == "transfer":
        xml = _twiml.transfer(prompt, target or "+10000000000")
    elif action == "voicemail":
        xml = _twiml.voicemail(prompt)
    elif action == "hangup":
        xml = _twiml.say_and_hangup(prompt)
    else:
        xml = _twiml.gather_response(prompt)

    return Response(content=xml, media_type="application/xml")
