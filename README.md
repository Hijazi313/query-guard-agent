# Query Sanitizer — Agentic Validation Pipeline

A self-correcting LangGraph agent that extracts, validates, and auto-corrects structured location queries using a multi-strategy correction loop.

---

## What it does

Takes a raw natural language query — potentially misspelled or ambiguous — and returns a clean, validated data structure with a confirmed city, country, and intent.

```
Input:  "Tell me weather in Duabi"
Output: { city: "dubai", country: "uae", intent: "weather", corrections: [...] }
```

The agent doesn't just extract. If extraction fails validation, it attempts correction through two strategies before giving up — fuzzy string matching first, LLM-based inference second.

---

## Architecture

The pipeline is a directed LangGraph graph with conditional routing:

```text
extractor → validator → [corrector → increment_retry → validator → ...]
    ↘             ↘ [llm_correction → increment_retry → validator → ...]
   hitl             ↘ [llm_city_corrector_from_list → hitl]
    ↓
process_hitl → validator
```

**Nodes:**

- `extractor` — structured LLM extraction via `with_structured_output` + Pydantic schema
- `validator` — deterministic rule checks against a known city/country dataset
- `corrector` — fuzzy string matching with deterministic similarity scoring
- `llm_correction` — general LLM-based city name inference for semantic errors
- `llm_city_corrector_from_list` — constrained LLM fallback using top-10 fuzzy candidates
- `increment_retry` — guards against infinite loops (MAX_RETRY = 2)
- `hitl` — evaluates rules to prepare the graph state for human intervention
- `process_hitl` — consumes human input, resets retries, and loops back into validation

The graph routes conditionally based on validation outcome and retry count. Validated results exit early; failed extractions exhaust retries then exit cleanly.

---

## Project Structure

This project follows a professional, modular architecture designed for scalability (May 2026 standards):

```text
.
├── agent/                  # Core LangGraph agent package
│   ├── state.py            # TypedDicts and schemas (SanitizerState, ExtractedInfo)
│   ├── config.py           # Env loading, LLM initialization, and dataset loading
│   ├── prompts.py          # ChatPromptTemplates and system prompts
│   ├── nodes/              # Node functions (pure state mutators)
│   │   ├── extraction.py
│   │   ├── validation.py
│   │   ├── correction.py
│   │   └── hitl.py
│   ├── routers.py          # Pure routing logic (typed decisions, no side-effects)
│   └── graph.py            # StateGraph builder and compilation
├── api/                    # [Future] REST layer (FastAPI)
├── evals/                  # [Future] Evaluation layer (RAGAS/LangSmith)
├── ui/                     # [Future] Next.js 15 Frontend
├── cities.json             # Static dataset
└── main.py                 # CLI entrypoint and human-in-the-loop runner
```

---

## Architectural Guidelines (L6 Critique Standards)

1. **Pure Routers**: Routing functions must be pure. They only read state, return typed strings (Literal edges), and have zero side effects (no API calls, no logging, no state mutation).
2. **Nodes as Mutators**: Nodes are responsible for taking the current state and returning a dictionary of updates.
3. **State Integrity**: The `retry_count` is reset whenever a human intervenes to ensure subsequent autocorrect attempts have a full retry budget for the new input.
4. **State Bleeding Prevention**: Unique `thread_id` generation per session prevents session state from bleeding into concurrent or future runs.

---

## Stack

| Layer             | Tool                                |
| ----------------- | ----------------------------------- |
| Orchestration     | LangGraph                           |
| LLM               | GPT-4o-mini (OpenAI)                |
| Structured output | Pydantic + `with_structured_output` |
| Fuzzy matching    | `difflib.get_close_matches`         |
| State management  | `TypedDict` + LangGraph state       |
| Package Manager   | UV (Fast Python package installer)  |
| Logging           | Structlog                           |
| Observability     | LangSmith (Tracing & Debugging)     |

---

## Key design decisions

**Why two correction strategies?**
Fuzzy matching is cheap and fast — it handles typos like "Duabi" → "dubai" with no LLM call. The LLM fallback handles phonetic or creative misspellings that difflib misses. Ordering them this way keeps cost low and only escalates when needed.

**Why `MAX_RETRY = 2`?**
Prevents runaway loops. The graph is designed to fail gracefully — a bad query exits with `validated: False` rather than crashing or looping forever.

**Why keep validation deterministic?**
The validator checks against a static dataset, not the LLM. This makes failures predictable and testable. LLMs are only used for correction, not judgment.

---

## Run it

```bash
# Clone the repository
git clone https://github.com/Hijazi313/QueryGuard-SanitizeAgent
cd QueryGuard-SanitizeAgent

# Configure environment
cp .env.example .env
# Edit .env and add:
# OPENAI_API_KEY=your_key
# LANGCHAIN_API_KEY=your_key (optional but recommended)

# Install dependencies and run (using uv)
uv sync
uv run main.py
```

## Observability

This project is integrated with **LangSmith** for full-trace observability. To enable tracing, ensure `LANGCHAIN_TRACING_V2=true` and your `LANGCHAIN_API_KEY` are set in the `.env` file. This allows you to inspect the agent's decision-making process, LLM inputs/outputs, and graph transitions in real-time.

---

## Project status

Core pipeline is stable. Planned next:

- Broader city dataset
- RAGAS-style evaluation layer
- REST endpoint wrapper

---

## Roadmap

### Done

- [x] Structured LLM extraction with Pydantic schema
- [x] Deterministic validation against static city/country dataset
- [x] Fuzzy string correction with calibrated similarity scoring
- [x] LLM-based city inference (General & List-constrained fallbacks)
- [x] **Confidence-aware routing** (High/Medium/Low buckets)
- [x] **Graph Optimization**: Removed redundant nodes, moved state updates to source
- [x] Retry counter guard (MAX_RETRY) to prevent infinite loops
- [x] Graceful failure — pipeline never crashes, exits with error state
- [x] UV Project Migration (Modern, stable package management)
- [x] **Human-in-the-loop (HITL)** — stateful pauses via `MemorySaver` and `interrupt_before`
- [x] **Modular Architecture** — production-grade Separation of Concerns (SoC) structure

---

### Planned

#### Real geocoding API integration

Replace the static cities.json dataset with a live geocoding call
(Google Maps, Nominatim, or Positionstack).

Benefits:

- Handles any city worldwide, not just the hardcoded list
- Returns lat/lng, timezone, and region — richer structured output
- Makes CITY_NOT_FOUND errors nearly impossible for real cities

The validator node would call the API as a confirmation step, not a
correction step — keeping the separation of concerns intact.

#### Evaluation layer

Add a small labeled test set and run RAGAS-style metrics:

- Extraction accuracy (did it get the right city/country/intent?)
- Correction precision (did corrections improve or worsen the result?)
- False positive rate (valid queries incorrectly flagged as errors?)

Goal: make the pipeline's reliability measurable, not just observable.

#### Expose as an API endpoint

Wrap the graph in a FastAPI route so it can be called from any service:

```
POST /sanitize
{ "query": "Tell me weather in Duabi" }

→ { "city": "dubai", "country": "uae", "intent": "weather", ... }
```

This is the step that makes it a reusable component, not just a script.

---

## Author

Muhammad Hmaza — backend & AI engineer.
Building in public. Feedback welcome.
