# Prompt log – Project 12

Chronological record of the prompts given to the AI assistant while building the project,
in the same "Message N" format as the Lab1 log.

---

### Message 1: Project analysis, roadmap and architecture

> Deeply analyze the project abstract (`tmp.md`), create a roadmap and designs for the project,
> acting as a senior software architect.

Outcome: `ROADMAP.md`, `ARCHITECTURE.md`, repository scaffold (Phase 0).

---

### Message 2: Phase 1 implementation (data pipeline)

> Alright, we got requirements and an architecture specification so let's start to implement the
> system; in case some data are needed generate synthetic data but in very small volume. Crucial
> is to check how algorithms work; the current implementation scope is Phase 1 from `ROADMAP.md`.
> After work is done, then update also Project documentation.

Outcome: `src/fraud_ae/data.py` (extract + sha256, dedupe, `log1p(Amount)`, hour-of-day sin/cos,
seeded legit-only split, train-only `StandardScaler`, `data_report.json`), `src/fraud_ae/synthetic.py`
(2 050-row creditcard-like frame shared by the tests and `fraud-ae prepare --synthetic`), EDA
figures in `plots.py`, `prepare` wired in the CLI, 19 Phase 1 tests (split invariants, leakage
controls, determinism, artifact round-trip). Run on the real dataset: 1 081 duplicates dropped,
473 frauds, split 169 952 / 56 887 / 56 887; sha256 pinned in all configs. README, ROADMAP and
ARCHITECTURE updated with the observed numbers.

---
