"""Shared Lightning AI / Nemotron Ultra client.

Reads auth from env (never hardcodes the key): LIGHTNING_API_KEY,
LIGHTNING_TEAMSPACE, LIGHTNING_CLOUD_PROJECT_ID, LIGHTNING_ORG.

Pinned model string (declared to organizers). Nemotron via litai exposes no
temperature knob -> noted as residual non-determinism. Defensive JSON
extraction (no schema enforcement available on this provider).
"""
import json
import os
import re

MODEL = os.environ.get("LITAI_MODEL", "lightning-ai/deepseek-v4-pro")
REQUIRED_ENV = ("LIGHTNING_API_KEY", "LIGHTNING_TEAMSPACE", "LIGHTNING_CLOUD_PROJECT_ID")


_CLIENT = None


def get_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"missing env vars: {missing}")
    from litai import LLM
    _CLIENT = LLM(model=MODEL, api_key=os.environ["LIGHTNING_API_KEY"])
    return _CLIENT


def chat(prompt, max_tokens=16000, reasoning_effort=None):
    """Returns (content, prompt_tokens, completion_tokens, finish_reason).
    reasoning_effort is only forwarded when explicitly set (some models don't
    accept it) — 'none' suppresses extended thinking on reasoning models."""
    kwargs = {"full_response": True}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    llm = get_client()
    resp = llm.chat(prompt, max_tokens=max_tokens, **kwargs)
    content = resp.choices[0].delta.content or ""
    finish_reason = resp.choices[0].finish_reason
    prompt_tokens = int(resp.usage.prompt_tokens)
    completion_tokens = int(resp.usage.completion_tokens)
    return content, prompt_tokens, completion_tokens, finish_reason


def extract_json_array(text):
    """Parse a raw JSON array out of model text (fences / prose / trailing junk).
    Tolerates trailing commas (a common LLM JSON bug)."""
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
    text = re.sub(r",\s*([\]}])", r"\1", text)  # strip trailing commas
    return json.loads(text)
