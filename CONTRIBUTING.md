# Contributing to TUP Detection

Thanks for your interest in extending TUP Detection. This project came out of the
**[Apart Research · Global South 2026](https://apartresearch.com/project/tup-detection-hybrid-promptinjection-guard-for-ai-generative-security-monitoring-r4w6)**
sprint, and we welcome contributions from the AI-safety and LLM-security community —
new detection rules, additional benchmarks, evaluation improvements, and bug fixes.

## Quick start (no credentials needed)

You can run and extend **Layer 1** (the deterministic regex engine) with zero secrets:

```bash
python3 -m venv .venv-pint && source .venv-pint/bin/activate
pip install -r scripts/requirements-pint.txt
pip install -r tup-manager/requirements.txt

python scripts/smoke_l1.py        # 30-second L1 smoke test, no HF token required
```

Sentinel v2 (Layer 2) and the optional L3 judge need credentials — see the
[README → Getting Started](README.md#getting-started).

## Adding a new Layer 1 rule

Detection rules live in [`policies/rules/`](policies/rules/) as one YAML file per rule
(`tup-rule-XXXX.yml`). To add one:

1. Copy an existing rule (e.g. `policies/rules/tup-rule-0001.yml`) to the next free id.
2. Fill in the fields:
   - `id`, `title`, `description`, `enabled: true`, `level` (0–15 severity).
   - `behavior` — one of the known behaviour labels (`prompt_injection`,
     `authority_spoofing`, `context_discovery`, `safety_bypass`, `pii_exfiltration`,
     `data_exposure`, `toxic_output`).
   - `match.field` (usually `prompt`) and `match.patterns` — Python regex strings.
     Prefer case-insensitive `(?i)` patterns and keep them tight to avoid false positives.
   - `framework_mapping` — always map to an `owasp_llm` category (e.g. `LLM01`), plus
     MITRE ATLAS / NIST AI RMF / EU AI Act where applicable. This OWASP mapping is what
     makes Layer 1 catches explainable, so it is required.
   - `response.alert_title` and `response.action`.
3. Add a benign **and** an attack example to `scripts/smoke_l1.py` (or a unit test) so
   the rule is exercised and we guard against false positives.
4. Run the smoke test and the suite (below) before opening a PR.

> Tip: a good rule earns its place by catching attacks Sentinel v2 misses, **without**
> flagging benign inputs. See the complementarity analysis in the README.

## Running the tests before a PR

```bash
python scripts/smoke_l1.py          # fast, no credentials
pytest tup-manager/tests/ -v        # unit test suite
```

Both must pass. If your change affects benchmark numbers, regenerate the evidence
notebook (`notebooks/tup_detection_guard_benchmark_report.ipynb`) so the committed
outputs stay in sync.

## Pull request guidelines

- Branch off `main`; keep PRs focused on a single rule/feature/fix.
- Describe **what** the change detects/fixes and **why**, with an example prompt.
- Never commit secrets — `.env` is git-ignored; use `notebooks/.env.pint.example` as the
  template.
- For new attack categories, link the relevant OWASP LLM / MITRE ATLAS reference.

## Reporting issues

Open a GitHub issue with a minimal reproducible prompt and the observed vs. expected
verdict. Security-sensitive findings: please contact the maintainers privately first.
