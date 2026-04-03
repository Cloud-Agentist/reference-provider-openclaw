"""
reference-provider-openclaw
----------------------------
Reference cognition provider using the OpenAI Agents SDK (openai-agents).
Accepts PerceptionFrame-based requests with time-sliced multimodal sensory data.
Returns motor commands (move, speak, gesture, act) parsed from text output.

This is a REQUEST-RESPONSE provider — the platform calls it on a heartbeat.

Port: 8082 (default)

Env vars:
  OPENAI_API_KEY  — required
  OPENAI_MODEL    — model name (default gpt-4o-mini)
  PORT            — override listen port
"""

import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agents import Agent, Runner, set_default_openai_key  # openai-agents SDK

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
PORT = int(os.environ.get("PORT", "8082"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

set_default_openai_key(OPENAI_API_KEY)


# ── Pydantic models ─────────────────────────────────────────────────────────

class SensoryChannel(BaseModel):
    facultyId: str
    modality: str
    payload: dict[str, Any]
    sources: list[dict[str, Any]] | None = None


class SensorySlice(BaseModel):
    sliceId: str
    actorId: str
    capturedAt: str
    durationMs: int
    channels: list[SensoryChannel]


class PerceptionFrame(BaseModel):
    frameId: str
    actorId: str
    capturedAt: str
    slices: list[SensorySlice]
    memoryContext: list[dict[str, Any]] | None = None
    selfState: dict[str, Any] | None = None
    attentionHints: list[str] | None = None


class MotorCommand(BaseModel):
    commandType: str
    actorId: str
    move: dict[str, Any] | None = None
    speak: dict[str, Any] | None = None
    gesture: dict[str, Any] | None = None
    act: dict[str, Any] | None = None


class ActorContextMeta(BaseModel):
    memories: list[dict[str, Any]] | None = None
    world: dict[str, Any] | None = None
    model_config = {"extra": "allow"}


class ActorContext(BaseModel):
    actorId: str | None = None
    actorType: str | None = None
    displayName: str | None = None
    activeGoals: list[str] | None = None
    sessionId: str | None = None
    metadata: ActorContextMeta | None = None


class ReasoningRequest(BaseModel):
    actorId: str
    perceptionFrame: PerceptionFrame | None = None
    input: str | None = None
    mode: str | None = "ask"
    requestId: str | None = None
    actorContext: ActorContext | None = None
    availableCapabilities: list[dict[str, str]] | None = None
    model_config = {"extra": "allow"}


class ReasoningResult(BaseModel):
    text: str
    requestId: str | None = None
    motorCommands: list[dict[str, Any]] | None = None
    proposedIntents: list[dict[str, Any]] | None = None
    providerMetadata: dict[str, Any] | None = None


# ── Perception to prompt ─────────────────────────────────────────────────────

def perception_to_prompt(frame: PerceptionFrame, direct_input: str | None = None) -> str:
    parts: list[str] = []

    if frame.selfState:
        s = frame.selfState
        lines: list[str] = []
        if s.get("status"):
            lines.append(f"Status: {s['status']}")
        if s.get("faculties"):
            lines.append(f"Faculties: {', '.join(s['faculties'])}")
        if s.get("currentGoals"):
            lines.append(f"Goals:\n" + "\n".join(f"  - {g}" for g in s["currentGoals"]))
        if lines:
            parts.append("## Your State\n" + "\n".join(lines))

    if frame.attentionHints:
        parts.append("## Attention\n" + "\n".join(f"- {h}" for h in frame.attentionHints))

    if frame.slices:
        parts.append(f"## Sensory Input ({len(frame.slices)} slices)")
        for sl in frame.slices:
            try:
                t = datetime.fromisoformat(sl.capturedAt.replace("Z", "+00:00")).strftime("%H:%M:%S")
            except Exception:
                t = sl.capturedAt
            ch_descs: list[str] = []
            for ch in sl.channels:
                try:
                    data = json.loads(ch.payload["data"]) if ch.payload.get("format") == "json" else ch.payload.get("data", "")
                    ch_descs.append(f"[{ch.modality}] {json.dumps(data) if isinstance(data, (dict, list)) else data}")
                except Exception:
                    ch_descs.append(f"[{ch.modality}] (data)")
            if ch_descs:
                parts.append(f"### {t}\n" + "\n".join(ch_descs))

    if frame.memoryContext:
        mem_lines = []
        for m in frame.memoryContext:
            content = m.get("content", {})
            text = content.get("text") if isinstance(content, dict) else str(content)
            mem_lines.append(f"- {text or json.dumps(m)}")
        parts.append("## Memories\n" + "\n".join(mem_lines))

    if direct_input:
        parts.append(f'## Direct Input\nThe user says: "{direct_input}"')

    return "\n\n".join(parts)


# ── System prompt ─────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = (
    "You are an embodied agent in a 3D virtual world. You perceive the world through "
    "time-sliced sensory data and can act through motor commands.\n\n"
    "When you want to act, describe your intended actions using this exact format (one per line):\n"
    "  [MOVE] area=commons\n"
    "  [SPEAK] content=Hello there! volume=normal\n"
    "  [GESTURE] type=wave\n"
    '  [ACT] action=calendar.event.create parameters={"title":"Meeting"} rationale=User requested a meeting\n\n'
    "Always provide natural language text alongside any action commands.\n\n"
)

MODE_PROMPTS = {
    "ask": "The user is asking a direct question. Answer concisely.",
    "plan": "Produce a step-by-step action plan.",
    "reflect": "Reflect on the situation with observations and recommendations.",
    "react": "React autonomously to your sensory input. If nothing interesting, observe briefly.",
}


def build_instructions(req: ReasoningRequest) -> str:
    mode = req.mode or "ask"
    parts = [BASE_SYSTEM_PROMPT + MODE_PROMPTS.get(mode, MODE_PROMPTS["ask"])]

    ctx = req.actorContext
    if ctx:
        if ctx.displayName:
            parts.append(f"You are speaking with {ctx.displayName}.")
        if ctx.activeGoals:
            goals = "\n".join(f"- {g}" for g in ctx.activeGoals)
            parts.append(f"Goals:\n{goals}")
        if ctx.metadata and ctx.metadata.memories:
            mem_lines = []
            for m in ctx.metadata.memories:
                content = m.get("content", {})
                text = content.get("text") if isinstance(content, dict) else str(content)
                mem_lines.append(f"- [{m.get('memory_type', 'memory')}] {text or str(m)}")
            parts.append("Memories:\n" + "\n".join(mem_lines))

    if req.availableCapabilities:
        cap_lines = [f"- {c['action']} [{c.get('sensitivityLevel', 'normal')}]: {c.get('description', '')}" for c in req.availableCapabilities]
        parts.append("Available actions:\n" + "\n".join(cap_lines))

    return "\n\n".join(parts)


# ── Parse motor commands from text ────────────────────────────────────────────

PARAM_RE = re.compile(r"(\w+)=(\{[^}]*\}|[^\s]+)")

MOVE_RE = re.compile(r"^\[MOVE\]\s*(.*)", re.IGNORECASE)
SPEAK_RE = re.compile(r"^\[SPEAK\]\s*(.*)", re.IGNORECASE)
GESTURE_RE = re.compile(r"^\[GESTURE\]\s*(.*)", re.IGNORECASE)
ACT_RE = re.compile(r"^\[ACT\]\s*(.*)", re.IGNORECASE)


def parse_params(param_str: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in PARAM_RE.finditer(param_str)}


def parse_motor_commands(text: str, actor_id: str) -> tuple[list[dict[str, Any]], str]:
    commands: list[dict[str, Any]] = []
    clean_lines: list[str] = []

    for line in text.split("\n"):
        trimmed = line.strip()

        m = MOVE_RE.match(trimmed)
        if m:
            p = parse_params(m.group(1))
            commands.append({
                "commandType": "move",
                "actorId": actor_id,
                "move": {"target": {k: v for k, v in p.items() if k in ("area", "targetActorId")}, "speed": p.get("speed", "walk")},
            })
            continue

        m = SPEAK_RE.match(trimmed)
        if m:
            p = parse_params(m.group(1))
            commands.append({
                "commandType": "speak",
                "actorId": actor_id,
                "speak": {"content": p.get("content", ""), "volume": p.get("volume", "normal")},
            })
            continue

        m = GESTURE_RE.match(trimmed)
        if m:
            p = parse_params(m.group(1))
            commands.append({
                "commandType": "gesture",
                "actorId": actor_id,
                "gesture": {"type": p.get("type", "idle")},
            })
            continue

        m = ACT_RE.match(trimmed)
        if m:
            p = parse_params(m.group(1))
            params = {}
            if "parameters" in p:
                try:
                    params = json.loads(p["parameters"])
                except json.JSONDecodeError:
                    pass
            commands.append({
                "commandType": "act",
                "actorId": actor_id,
                "act": {"action": p.get("action", ""), "parameters": params, "rationale": p.get("rationale", "")},
            })
            continue

        clean_lines.append(line)

    return commands, "\n".join(clean_lines).strip()


# ── FastAPI app ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


app = FastAPI(title="reference-provider-openclaw", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"ok": True, "service": "reference-provider-openclaw", "model": OPENAI_MODEL, "interactionPattern": "request-response"}


@app.post("/reasoning", response_model=ReasoningResult)
async def reasoning(req: ReasoningRequest):
    instructions = build_instructions(req)

    # Build user message from perception frame and/or direct input
    if req.perceptionFrame:
        user_input = perception_to_prompt(req.perceptionFrame, req.input)
    elif req.input:
        user_input = req.input
    else:
        user_input = "(no input)"

    agent = Agent(
        name="openclaw",
        instructions=instructions,
        model=OPENAI_MODEL,
    )

    start = time.monotonic()
    try:
        result = await Runner.run(agent, user_input)
        raw_text = result.final_output or ""
        duration_ms = int((time.monotonic() - start) * 1000)

        # Parse motor commands from text
        motor_commands, clean_text = parse_motor_commands(raw_text, req.actorId)

        session_id = req.actorContext.sessionId if req.actorContext else None

        return ReasoningResult(
            text=clean_text,
            requestId=req.requestId,
            motorCommands=motor_commands if motor_commands else None,
            providerMetadata={
                "provider": "openclaw",
                "model": OPENAI_MODEL,
                "mode": req.mode or "ask",
                "durationMs": duration_ms,
                "motorCommandCount": len(motor_commands),
                **({"sessionId": session_id} if session_id else {}),
            },
        )
    except Exception as exc:  # noqa: BLE001
        duration_ms = int((time.monotonic() - start) * 1000)
        return JSONResponse(
            status_code=502,
            content={
                "error": "Inference failed",
                "detail": str(exc),
                "durationMs": duration_ms,
            },
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
