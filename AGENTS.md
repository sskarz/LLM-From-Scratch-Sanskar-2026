# AGENTS.md

## Repo overview
- **What this is:** A learning exercise building a GPT model from scratch (PyTorch). Not production code.
- **Entry point:** `chapter_4/main.py` — creates a DummyGPTModel and runs inference on two short strings.
- **Supporting code:** `chapter_4/gpt.py` (config constants), `chapter_4/DummyGPTModel.py` (model + stub transformer blocks), `attn_mechanisms/self_attn.py` (self-attention demo).

## Setup
```bash
uv sync          # install deps from pyproject.toml (torch, numpy, tiktoken)
uv run python chapter_4/main.py   # run the model
```

- Python 3.14 required (`.python-version`).
- No test/lint/typecheck setup exists yet.

## Running code
```bash
uv run python main.py              # root stub (prints hello)
uv run python chapter_4/main.py   # actual GPT model demo
```

## Constraints
- No existing test framework, linter, or formatter. Don't assume `pytest` or `ruff` are available — install them first if needed.
- The model uses MPS on Mac (`torch.backends.mps.is_available()`). Falls back to CPU.
- `DummyTransformerBlock` and `DummyLayerNorm` are no-op stubs — the model runs but doesn't actually transform anything.
