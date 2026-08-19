#!/usr/bin/env python3
"""
Meeting Notes -> Action Items Agent
------------------------------------
Takes a meeting transcript (plain text) and produces:
  1. A structured summary (decisions + discussion points)
  2. A list of action items (owner, due date where stated)

Works against any OpenAI-compatible chat endpoint, which covers:
  - Ollama running locally (default)
  - LM Studio
  - OpenAI's real API
  - Groq

See README.md for setup.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = """You are a precise meeting-notes assistant.

You will be given a raw meeting transcript. Extract information and return
ONLY a single valid JSON object (no markdown fences, no commentary) matching
this exact schema:

{
  "summary": {
    "decisions": [string, ...],
    "discussion_points": [string, ...]
  },
  "action_items": [
    {
      "item": string,
      "owner": string or null,
      "due_date": string or null
    }
  ]
}

Rules:
- "decisions" are things the group explicitly agreed on or decided.
- "discussion_points" are notable topics raised that were NOT decided or
  assigned as action items (e.g. open questions, deferred topics).
- An action item must be tied to a clear commitment or assignment
  ("X will...", "X to...", "I'll get...", "can you..." + agreement).
  Do not invent action items that weren't actually committed to.
- "owner" is the person's name as stated. If no owner is clearly stated,
  use null. Do not guess who is responsible.
- "due_date" is the due date AS STATED in the transcript (e.g. "August 22",
  "end of week"). Do not resolve relative dates ("end of week") to a
  calendar date — just report what was said. If no due date was mentioned,
  use null.
- Do not include any text outside the JSON object.
"""


def get_client() -> OpenAI:
    """Build an OpenAI-compatible client. Defaults to a local Ollama server."""
    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    api_key = os.environ.get("LLM_API_KEY", "ollama")  # Ollama ignores the key
    return OpenAI(base_url=base_url, api_key=api_key)


def call_llm(client: OpenAI, model: str, transcript: str, strict: bool = False) -> str:
    user_msg = f"Meeting transcript:\n\n{transcript}"
    if strict:
        user_msg += "\n\nReturn ONLY the JSON object. No markdown, no explanation."
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def extract_json(text: str) -> dict:
    """Parse JSON from the model output, tolerating stray text/markdown fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: pull the first {...} block out of the response
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


def validate(data: dict) -> None:
    assert "summary" in data, "missing 'summary'"
    assert "decisions" in data["summary"], "missing 'summary.decisions'"
    assert "discussion_points" in data["summary"], "missing 'summary.discussion_points'"
    assert "action_items" in data, "missing 'action_items'"
    assert isinstance(data["action_items"], list), "'action_items' must be a list"
    for i, item in enumerate(data["action_items"]):
        assert "item" in item, f"action_items[{i}] missing 'item'"
        item.setdefault("owner", None)
        item.setdefault("due_date", None)


def save_outputs(data: dict, outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "summary.json", "w") as f:
        json.dump(data, f, indent=2)

    with open(out / "action_items.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "owner", "due_date"])
        for ai in data["action_items"]:
            writer.writerow([ai.get("item", ""), ai.get("owner") or "", ai.get("due_date") or ""])


def run(transcript_path: str, output_dir: str, model: str) -> dict:
    transcript = Path(transcript_path).read_text()
    client = get_client()

    raw = call_llm(client, model, transcript)
    try:
        data = extract_json(raw)
        validate(data)
    except (ValueError, AssertionError, json.JSONDecodeError) as e:
        print(f"[warn] First response failed to parse/validate ({e}). Retrying once...", file=sys.stderr)
        raw = call_llm(client, model, transcript, strict=True)
        data = extract_json(raw)
        validate(data)

    save_outputs(data, output_dir)
    return data


def main():
    parser = argparse.ArgumentParser(description="Meeting Notes -> Action Items Agent")
    parser.add_argument("--transcript", required=True, help="Path to a transcript .txt file")
    parser.add_argument("--output", default="output", help="Directory to write results to")
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL", "llama3.1"),
        help="Model name as known to your LLM server (default: env LLM_MODEL or 'llama3.1')",
    )
    args = parser.parse_args()

    data = run(args.transcript, args.output, args.model)

    print(json.dumps(data, indent=2))
    print(f"\nSaved: {args.output}/summary.json and {args.output}/action_items.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
