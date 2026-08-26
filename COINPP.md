# CoIN++ on MCITlib

This integration runs the factor-controlled CoIN++ streams with MCITlib's
LLaVA-1.5 continual-learning methods. It does not copy the image corpus.
Prepared JSON files and symlinks live under `datasets/CoIN++`.

## Dataset profiles

The recommended clean source is expected at:

```text
/home/chencheng/data/Code/Easy_Train_MLLM/cl_dataset/coin_factor1_final_clean
```

The historical `strict` and `legacy_10k` profiles use:

```text
/home/chencheng/data/Code/Easy_Train_MLLM/cl_dataset/coin_factor1_final
```

The image root is expected at:

```text
/home/chencheng/data/Code/Easy_Train_MLLM/cl_dataset
```

Three profiles are supported:

| Profile | Train/eval per category | Image policy | Intended use |
|---|---:|---|---|
| `clean_5k` | 5,000 / 1,000 | Every row has a real image; IconQA is excluded | Recommended method comparison |
| `strict` | 4,000 / 500 | Image-backed subset of the legacy source | Compatibility baseline |
| `legacy_10k` | 10,000 / 1,000 | Preserves source text-only/missing-image rows | Historical reproduction only |

The clean source is derived from the TextVQA-repaired final pool, excludes all
IconQA records, and removes every row without a resolvable image. The legacy
10k source contains text-only MMMU, SLAKE, and retained IconQA records. Use
`clean_5k` for comparable visual continual-learning results.

All clean JSON files use the consolidated `coin_factor1_final_clean/images/`
prefix. The complete distribution and image validation report is available at
`datasets/CoIN++/media/coin_factor1_final_clean/final_stats/factor1_final_distribution.md`.

Clean-profile totals:

| Factor stream | Stages | Train | Eval |
|---|---:|---:|---:|
| visual_substrate | 6 | 30,000 | 6,000 |
| skill_requirement | 10 | 50,000 | 10,000 |
| evidence_complexity | 4 | 20,000 | 4,000 |

Prepare or refresh the local views:

```bash
cd /home/chencheng/data/Code/MCITlib
python3 scripts/coinpp/prepare_coinpp.py --profile clean_5k
```

If another server only has the self-contained dataset under
`datasets/CoIN++/media/coin_factor1_final_clean`, prepare everything locally:

```bash
bash scripts/coinpp/prepare_from_local_media.sh
```

This generates the method-facing `train.json`, `questions.json`, `replay.json`,
`router.json`, and data configs without requiring an Easy_Train_MLLM checkout.

Use `--source-root` and `--media-root` if the Easy_Train_MLLM checkout
moves. Data configs are generated under
`configs/data_configs/CoIN++/<profile>/<factor>/`.

## Model configuration

The default local model file is
`configs/model_configs/llava_coinpp_local.json`. It references the Vicuna
base model, LLaVA projector, and CLIP vision tower already present in the
Easy_Train_MLLM checkout. Update this JSON if those checkpoints move.

Run a no-GPU command validation first:

```bash
DRY_RUN=1 PROFILE=clean_5k METHOD=LoRA-FT FACTOR=evidence_complexity \
  bash scripts/coinpp/run_method_pipeline.sh
```

## Train, evaluate, and score

A complete single-method pipeline is:

```bash
METHOD=LoRA-FT \
FACTOR=evidence_complexity \
PROFILE=clean_5k \
GPU_NUM=4 \
EVAL_GPU=0 \
  bash scripts/coinpp/run_method_pipeline.sh
```

The pipeline runs three independent phases:

1. Sequential training with the selected method's own MCITlib implementation.
2. One model load per stage, followed by evaluation on every factor category.
3. Matrix aggregation with native-proxy, normalized-exact, and token-F1 tracks.

Use `MODE=train`, `MODE=eval`, or `MODE=score` to run one phase.
`START_STAGE` and `STOP_STAGE` select a stage range. Completed
training stages are marked by `.coinpp_complete.json` and skipped safely unless
`FORCE=1` is set. Completed prediction cells are also skipped by row count.

Run several methods and factors sequentially:

```bash
METHODS="LoRA-FT Replay OLoRA MoELoRA CL-MoE HiDe DISCO" \
FACTORS="visual_substrate skill_requirement evidence_complexity" \
PROFILE=clean_5k \
  bash scripts/coinpp/run_all_methods.sh
```

The default multi-method list is deliberately limited to the methods with no
extra preprocessing stage. A full all-factor run is expensive; start with
`evidence_complexity` and a one-stage smoke run.

## Method coverage

| Method | Training launcher | Matrix evaluation | Extra requirement |
|---|---|---|---|
| LoRA-FT | Automatic | Automatic | None |
| Replay | Automatic | Automatic | Cumulative replay JSON is generated |
| OLoRA | Automatic | Automatic | None |
| MoELoRA | Automatic | Automatic | None |
| ModalPrompt | Automatic | Automatic | Uses the configured CLIP tower/projector |
| CL-MoE | Automatic | Automatic | None |
| HiDe | Automatic | Automatic | Uses the configured CLIP text tower |
| RegLoRA | Automatic | Automatic on merged checkpoints | Expensive merge/key-element stages |
| DISCO | Automatic | Automatic | Uses the configured CLIP text tower |
| SMoLoRA | Automatic | Automatic | Factor instruction embeddings |
| KeepLoRA | Automatic | Automatic on merged checkpoints | Gradient extraction before/after every task |
| MR-LoRA | Expert/router training | Not in generic matrix runner | Factor-specific router dispatch |

Build SMoLoRA instruction embeddings before training:

```bash
python3 scripts/coinpp/build_smolora_embeddings.py \
  --factor evidence_complexity \
  --profile clean_5k \
  --model-path /path/to/all-MiniLM-L6-v2

SMOLORA_EMB="$PWD/datasets/CoIN++/embeddings/clean_5k/evidence_complexity_smolora.pkl" \
METHOD=SMoLoRA FACTOR=evidence_complexity PROFILE=clean_5k \
  bash scripts/coinpp/run_method_pipeline.sh
```

MR-LoRA router JSON is cumulative and balanced across all categories seen at
each stage. Its target is an expert letter, not the original VQA answer.
MCITlib's stock MR-LoRA inference uses benchmark-specific hard-coded expert
maps, so this integration does not claim a comparable routed matrix until a
CoIN++ factor-specific dispatch evaluator is supplied.

## Outputs

Training checkpoints:

```text
checkpoints/CoIN++/<profile>/<factor>/<method>/
```

Predictions and stage matrices:

```text
results/CoIN++/<profile>/<factor>/<method>/eval/
  1_<first_category>/predictions/<eval_category>.jsonl
  ...
  summary/matrix_native_proxy.csv
  summary/matrix_exact_match.csv
  summary/matrix_token_f1.csv
  summary/forgetting_native_proxy.csv
  summary/summary.json
  summary/summary.md
```

The local `native_proxy` uses the best metric supported by retained
annotations: multiple-choice accuracy, single-reference ANLS, relaxed numeric
accuracy, or normalized exact match. It is explicitly not called official when
the original benchmark annotations are unavailable.

## Independent standard and LLM-Judge tracks

The prediction schema is compatible with the existing Easy_Train_MLLM CoIN++
dual evaluator. The standard/native score and the all-sample LLM Judge score
remain independent.

On a server hosting an OpenAI-compatible judge:

```bash
METHOD=LoRA-FT \
FACTOR=evidence_complexity \
PROFILE=clean_5k \
PYTHON=/path/to/coin/bin/python \
JUDGE_MODEL=qwen3.6 \
JUDGE_BASE_URL=http://127.0.0.1:8001/v1 \
JUDGE_CACHE="$PWD/results/CoIN++/judge_cache/qwen3.6.jsonl" \
  bash scripts/coinpp/run_dual_eval.sh
```

The dual summary is written to `eval/summary_dual`. Judge cache is
append-only, so rerunning retries only missing predictions.

## Important experimental controls

Use the same profile, task order, base model, LoRA budget, epoch count, global
batch size, and evaluation set for every method. Report method-specific
parameter counts and wall-clock cost separately. Do not mix strict and legacy
rows in one comparison, and do not merge native/direct scores with Judge
scores.
