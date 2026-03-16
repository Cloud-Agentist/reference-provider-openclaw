"""
reference-provider-openclaw
----------------------------
Reference cognition provider using the OpenAI Agents SDK (openai-agents).
Implements POST /reasoning (ReasoningRequest → ReasoningResult) so it can be
registered in the cognition-provider-registry and compared against other
providers in the reasoning-gym.

Port: 8082 (default)

Env vars:
  OPENAI_API_KEY  — required
  OPENAI_MODEL    — model name (default gpt-4o-mini)
  PORT            — override listen port

Context enrichment (mirrors reference-provider claude mode):
  - actorContext.displayName          → injected into agent instructions
  - actorContext.activeGoals          → injected as bullet list
  - actorContext.metadata.memories    → recent memories injected
  - actorContext.metadata.world       → world facts injected

Intent proposal:
  Detects sensitive-action keywords in the input and proposes a structured
  intent in proposedIntents[]. Mirrors the stub provider behaviour so the
  reasoning-gym can exercise governance flows regardless of which provider
  is active.
"""

import os
import re
import time
import uuid
from contextlib import asynccontextmanager
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


# ── Pydantic models ───────────────────────────────────────────────────────────

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
    input: str
    mode: str | None = "ask"
    requestId: str | None = None
    actorContext: ActorContext | None = None
    model_config = {"extra": "allow"}


class ReasoningResult(BaseModel):
    text: str
    requestId: str | None = None
    confidence: float | None = None
    proposedIntents: list[dict[str, Any]] | None = None
    providerMetadata: dict[str, Any] | None = None


# ── Sensitive keyword detection ───────────────────────────────────────────────

SENSITIVE_PATTERNS = [
    (re.compile(r'\b(delete|remove|clear|wipe)\b.*(wishlist|list|cart|saved)', re.I),
     "wishlist.items.delete",
     "User requested deletion of wishlist — irreversible action requiring approval."),
    (re.compile(r'\b(buy|purchase|order|checkout)\b', re.I),
     "finance.purchase",
     "User requested a purchase — financial action requiring approval."),
    (re.compile(r'\b(transfer|send)\b.*(money|funds|\$|£|€|\d+\s*(usd|eur|gbp))', re.I),
     "finance.transfer",
     "User requested a financial transfer — requires approval."),
    (re.compile(r'\b(cancel|unsubscribe)\b.*(subscription|plan|membership)', re.I),
     "subscription.cancel",
     "User requested subscription cancellation — irreversible action requiring approval."),
]


def detect_sensitive_intent(actor_id: str, input_text: str) -> dict[str, Any] | None:
    for pattern, action, rationale in SENSITIVE_PATTERNS:
        if pattern.search(input_text):
            return {
                "intentId": str(uuid.uuid4()),
                "actorId": actor_id,
                "action": action,
                "sensitiveAction": True,
                "rationale": rationale,
                "confidence": 0.75,
            }
    return None


# ── System prompt construction ────────────────────────────────────────────────

BASE_PROMPTS = {
    "ask":     "You are a helpful assistant. Answer the user's question clearly and concisely.",
    "plan":    "You are a planning assistant. Break the user's goal into a clear, ordered set of steps. Think step by step.",
    "reflect": "You are a reflective assistant. Review the user's input and provide thoughtful observations, identifying strengths, risks, and improvements.",
}


def build_instructions(req: ReasoningRequest) -> str:
    base = BASE_PROMPTS.get(req.mode or "ask", BASE_PROMPTS["ask"])
    parts = [base]

    ctx = req.actorContext
    if not ctx:
        return base

    if ctx.displayName:
        parts.append(f"You are speaking with {ctx.displayName}.")
    if ctx.actorType:
        parts.append(f"Actor type: {ctx.actorType}.")
    if ctx.activeGoals:
        goals = "\n".join(f"- {g}" for g in ctx.activeGoals)
        parts.append(f"Actor's active goals:\n{goals}")

    meta = ctx.metadata
    if meta:
        if meta.memories:
            mem_lines = []
            for m in meta.memories:
                content = m.get("content", {})
                text = content.get("text") if isinstance(content, dict) else str(content)
                mem_type = m.get("memory_type", "memory")
                mem_lines.append(f"- [{mem_type}] {text or str(m)}")
            parts.append("Actor's recent memories:\n" + "\n".join(mem_lines))

        if meta.world:
            fact_lines = "\n".join(f"  {k}: {v}" for k, v in meta.world.items())
            parts.append(f"Current world context:\n{fact_lines}")

    return "\n\n".join(parts)


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


app = FastAPI(title="reference-provider-openclaw", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"ok": True, "service": "reference-provider-openclaw", "model": OPENAI_MODEL}


@app.post("/reasoning", response_model=ReasoningResult)
async def reasoning(req: ReasoningRequest):
    instructions = build_instructions(req)

    agent = Agent(
        name="openclaw",
        instructions=instructions,
        model=OPENAI_MODEL,
    )

    start = time.monotonic()
    try:
        result = await Runner.run(agent, req.input)
        text = result.final_output or ""
        duration_ms = int((time.monotonic() - start) * 1000)

        sensitive_intent = detect_sensitive_intent(req.actorId, req.input)
        proposed_intents = [sensitive_intent] if sensitive_intent else None
        session_id = req.actorContext.sessionId if req.actorContext else None

        return ReasoningResult(
            text=text,
            requestId=req.requestId,
            proposedIntents=proposed_intents,
            providerMetadata={
                "provider": "openclaw",
                "model": OPENAI_MODEL,
                "mode": req.mode or "ask",
                "durationMs": duration_ms,
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
