#!/usr/bin/env bash
# Tier B benchmark via HF Inference (no local model download).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv-pint/bin/activate
export PYTHONPATH="$ROOT/tup-manager:$ROOT/scripts:${PYTHONPATH:-}"
export INJECTION_CLASSIFIER_BACKEND=hf
export INJECTION_MODEL=rogue-security/prompt-injection-jailbreak-sentinel-v2
export BENIGN_GUARD_ENABLED=false

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${SENTINEL_API_KEY:-}${HF_TOKEN:-}${HUGGINGFACE_TOKEN:-}" ]]; then
  echo "ERROR: set SENTINEL_API_KEY in .env (Read token + accept Sentinel v2 license)" >&2
  exit 1
fi

if [[ -z "${HF_INFERENCE_ENDPOINT:-}" ]]; then
  echo "WARNING: HF_INFERENCE_ENDPOINT not set. Using serverless (may fail for Sentinel v2)" >&2
  echo "  Create dedicated endpoint at: https://ui.endpoints.huggingface.co/new" >&2
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Sentinel v2 Tier B Benchmark (deepset + HF Inference)"
echo "════════════════════════════════════════════════════════════════════"
echo ""

mkdir -p notebooks/data/external/results

echo "1️⃣  Verify HF Inference Endpoint..."
python scripts/verify_hf_endpoint.py || {
  echo "ERROR: HF Endpoint verification failed"
  exit 1
}
echo ""

echo "2️⃣  Import deepset (dataset only, no model weights)..."
python scripts/import_external_dataset.py \
  --preset deepset \
  --out notebooks/data/external/deepset.yaml
echo ""

echo "3️⃣  Benchmark deepset via HF Inference (Sentinel v2)..."
echo "    Mode: benchmark | Effective threshold: ${INJECTION_THRESHOLD_STRICT:-0.15}"
echo "    Warmup will run automatically (scale-to-zero cold start)"
echo ""
python scripts/run_pint_benchmark.py \
  --dataset notebooks/data/external/deepset.yaml \
  --detection-mode benchmark \
  --results-out notebooks/data/external/results/deepset-sentinel-hf.json

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "✅ Benchmark completed"
echo "📊 Results: notebooks/data/external/results/deepset-sentinel-hf.json"
echo ""
echo "Next steps:"
echo "  1. Review Overall Accuracy in results JSON"
echo "  2. Pause the inference backend when idle (HF Space or Endpoint UI)"
echo "  3. Re-run from cache (no API) with --scores-cache <results>.scores.json"
echo "  4. Run strict mode for comparison:"
echo "     python scripts/run_pint_benchmark.py \\"
echo "       --dataset notebooks/data/external/deepset.yaml \\"
echo "       --detection-mode strict \\"
echo "       --results-out notebooks/data/external/results/deepset-sentinel-strict.json"
echo "════════════════════════════════════════════════════════════════════"
