# Meeting Notes → Action Items Agent

Takes a meeting transcript and produces a structured summary (decisions +
discussion points) and a list of action items (owner, due date where stated).

**Input → Output:** *"My agent takes a raw meeting transcript (plain text) and
produces a structured JSON summary plus a CSV of action items with owners and
due dates."*

---

## 1. Setup

### Requirements
- Python 3.9+
- An OpenAI-compatible LLM endpoint. This defaults to a **local Ollama
  server**, so no API key or internet access is required.

### Install

```bash
git clone <this-repo-url>
cd meeting-notes-agent
pip install -r requirements.txt
```

### Configure your LLM

**Option A — Local Ollama (default, no API key needed)**

```bash
# Install Ollama from https://ollama.com if you don't have it
ollama pull llama3.1
ollama serve   # usually already running as a background service
```

No further config needed — the script defaults to
`http://localhost:11434/v1` with model `llama3.1`.

**Option B — OpenAI, Groq, or another OpenAI-compatible provider**

```bash
cp .env.example .env
# edit .env: set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL for your provider
export $(cat .env | xargs)   # or use a tool like python-dotenv / direnv
```

## 2. Run it

```bash
python meeting_agent.py --transcript data/sample_transcript.txt --output output
```

This prints the structured JSON to stdout and writes:
- `output/summary.json` — decisions, discussion points, and action items
- `output/action_items.csv` — action items as a flat table

To run on your own transcript, just point `--transcript` at a different
`.txt` file. Override the model with `--model <name>` if you don't want the
default.

## 3. Sample input / output

- Input: [`data/sample_transcript.txt`](data/sample_transcript.txt) — a
  4-person marketing sync with 3 decisions, 4 clear action items, and one
  deferred topic with no owner (used to test the model doesn't over-extract).
- Output (captured from a real run, included so you can inspect results
  without setting up an LLM first): [`output/summary.json`](output/summary.json),
  [`output/action_items.csv`](output/action_items.csv).

Re-running `meeting_agent.py` against the sample transcript with your own
local model will overwrite these with a fresh run.

## 4. Design choices

- **Single structured LLM call, not multi-step RAG.** A meeting transcript is
  short enough to fit entirely in context, so there's no retrieval step —
  the whole transcript goes in one prompt with a strict JSON schema. This
  keeps the agent fast and removes a class of retrieval bugs.
- **OpenAI-compatible client for any backend.** Using the `openai` Python SDK
  pointed at a configurable `base_url` means the same code runs against
  Ollama, LM Studio, Groq, or real OpenAI — just an env var change, no code
  change. This was chosen over an Ollama-specific SDK for portability.
- **Schema-constrained prompting + validation + one retry.** The system
  prompt pins an exact JSON shape. `extract_json()` tolerates minor
  formatting noise (e.g. the model wrapping output in ```` ```json ```` fences).
  `validate()` checks required keys exist before anything is saved. If the
  first response fails to parse or validate, the agent retries once with a
  stricter instruction rather than failing outright — this measurably
  improves reliability with smaller local models, which sometimes add
  commentary around the JSON.
- **Decision boundary for action items.** The prompt explicitly instructs
  the model to only extract action items tied to a clear commitment
  ("X will...", "I'll get..."), and to route open-but-unassigned topics into
  `discussion_points` instead. The sample transcript's influencer-partnership
  tangent is a deliberate test of this — it should NOT show up as an action
  item.

## 5. Tradeoffs & what I'd improve with more time

- **Relative due dates aren't resolved.** "End of week" is captured verbatim
  rather than converted to a calendar date, because doing that correctly
  requires knowing the meeting date and the team's week-start convention.
  With more time I'd pass the meeting date into the prompt and have the
  model (or a small dateutil-based post-processor) normalize relative dates
  to ISO 8601 where unambiguous.
- **No speaker diarization assumptions.** The agent assumes the transcript
  already has speaker labels (e.g. `Name: ...`). A raw, unlabeled transcript
  (e.g. from an ASR tool) would need a diarization or speaker-attribution
  step first — out of scope here but a natural next step alongside the
  Reception/Voice agent pattern.
- **Single-pass extraction, no self-critique step.** For longer or noisier
  transcripts, a second "review your own extraction against the transcript"
  pass would likely catch missed or hallucinated action items. Skipped here
  to keep the agent to one LLM call for speed and cost, given the 24-hour
  window.
- **Local models can be inconsistent with strict JSON.** The retry-once
  logic covers most cases in testing, but a production version would want a
  bounded retry loop (e.g. 3 attempts) and a fallback to returning the raw
  text with a `"parse_error": true` flag rather than crashing.
