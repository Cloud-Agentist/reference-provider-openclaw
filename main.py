"""
reference-provider-openclaw
----------------------------
Baseline reasoning provider built on the OpenAI Agents SDK (openai-agents).
Implements POST /reasoning (ReasoningRequest → ReasoningResult) so it can be
registered in the cognition-provider-registry and compared against Dreasen in
the reasoning-gym.

Port: 8082 (default)
"""

import os
import time
import asyncio
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


# ── Pydantic models (aligned with ReasoningRequest / ReasoningResult schemas) ─

class ActorContext(BaseModel):
    actorType: str | None = None
    displayName: str | None = None
    activeGoals: list[str] | None = None
    metadata: dict[str, Any] | None = None


class ReasoningRequest(BaseModel):
    actorId: str
    input: str
    mode: str | None = "ask"
    requestId: str | None = None
    actorContext: ActorContext | None = None


class ReasoningResult(BaseModel):
    text: str
    requestId: str | None = None
    confidence: float | None = None
    providerMetadata: dict[str, Any] | None = None


# ── System prompt construction ────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "ask": "You are a helpful assistant. Answer the user's question clearly and concisely.",
    "plan": "You are a planning assistant. Break the user's goal into a clear, ordered set of steps. Think step by step.",
    "reflect": "You are a reflective assistant. Review the user's input and provide thoughtful observations, identifying strengths, risks, and improvements.",
}


def build_instructions(req: ReasoningRequest) -> str:
    base = SYSTEM_PROMPTS.get(req.mode or "ask", SYSTEM_PROMPTS["ask"])
    parts = [base]

    ctx = req.actorContext
    if ctx:
        if ctx.displayName:
            parts.append(f"You are speaking with {ctx.displayName}.")
        if ctx.actorType:
            parts.append(f"Actor type: {ctx.actorType}.")
        if ctx.activeGoals:
            goals = "\n".join(f"- {g}" for g in ctx.activeGoals)
            parts.append(f"Actor's active goals:\n{goals}")

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

        return ReasoningResult(
            text=text,
            requestId=req.requestId,
            providerMetadata={
                "provider": "openclaw",
                "model": OPENAI_MODEL,
                "mode": req.mode or "ask",
                "durationMs": duration_ms,
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
