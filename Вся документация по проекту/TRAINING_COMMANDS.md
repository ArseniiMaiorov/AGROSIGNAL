# Training Commands

Commands are run from the project root:

```bash
cd /home/arsenii-maiorov/Documents/SUAI/Диплом/AutoDetect_v2.0
```

## Runtime / Release

Full clean start:

```bash
make fresh-start
```

Full audit:

```bash
make audit-full
```

Release smoke:

```bash
make release-smoke
```

Russian detect benchmark:

```bash
make detect-benchmark-ru
```

North / central / south QA summary:

```bash
make benchmark-north-south
```

## Training Dependency Install

Install CPU-safe training dependencies:

```bash
make install-training-deps
```

## One Command

Full one-shot pipeline:

```bash
make train-all-in-one
```

What it does:

- installs CPU-safe training dependencies
- imports local training files from `training_inputs/`
- reads Sentinel Hub credentials from `.env`
- uses automatic failover `primary -> reserv -> second_reserv`
- downloads open/public data
- prepares manifests and weak labels
- trains boundary candidate
- builds yield corpus
- trains yield baseline
- registers yield ensemble metadata
- runs QA summary / benchmark stage
- writes candidate manifest

Expected local files in `training_inputs/`:

```text
training_inputs/field_boundaries.geojson   (or .gpkg)
training_inputs/yield_history.csv
training_inputs/crop_plan.csv
training_inputs/soil_samples.csv           (optional)
training_inputs/management_events.csv      (optional)
training_inputs/weather_daily.csv          (optional)
```

Create templates for that folder:

```bash
make init-training-inputs
```

## Dry Run

Show the full training orchestration without launching heavy stages:

```bash
DRY_RUN=1 make train-orchestrated
```

Show individual stages:

```bash
DRY_RUN=1 make train-open-data-download
DRY_RUN=1 make train-boundary-v3-cpu
DRY_RUN=1 make train-yield-corpus
DRY_RUN=1 make train-yield-baseline
DRY_RUN=1 make train-yield-ensemble
```

## Production-Safe CPU Training

Default CPU-safe orchestrated pipeline:

```bash
BATCHED=1 LOW_MEM=1 EXPERIMENTAL_LSTM=0 BACKBONE=efficientnet_b0 make train-orchestrated
```

Equivalent one-shot command with explicit CPU-safe defaults:

```bash
BATCHED=1 LOW_MEM=1 EXPERIMENTAL_LSTM=0 BACKBONE=efficientnet_b0 make train-all-in-one
```

Default stages:

```text
download,prepare,train-boundary,benchmark,export,register
```

Run only selected stages:

```bash
STAGES=download,prepare,train-boundary,benchmark \
BATCHED=1 LOW_MEM=1 EXPERIMENTAL_LSTM=0 BACKBONE=efficientnet_b0 \
make train-orchestrated
```

Boundary + yield full local candidate cycle:

```bash
STAGES=download,prepare,train-boundary,train-yield-corpus,train-yield-baseline,train-yield-ensemble,benchmark,export,register \
BATCHED=1 LOW_MEM=1 EXPERIMENTAL_LSTM=0 BACKBONE=efficientnet_b0 \
make train-orchestrated
```

Promote only if benchmark gates are green:

```bash
make train-promote-if-green
```

## Stage Commands

Open/public corpus manifest + Sentinel fetch + weak labels:

```bash
make train-open-data-download
```

Import local CSV/GeoJSON training files into the app DB:

```bash
make train-import-data
```

Boundary v3 CPU candidate training:

```bash
make train-boundary-v3-cpu
```

Yield corpus from DB tables:

```bash
make train-yield-corpus
```

Yield baseline model:

```bash
make train-yield-baseline
```

Yield ensemble metadata / interval layer:

```bash
make train-yield-ensemble
```

## Sentinel Hub Failover

No manual credential switch is required.

The pipeline reads credentials from `.env` in this order:

1. `SH_CLIENT_ID` / `SH_CLIENT_SECRET`
2. `SH_CLIENT_ID_reserv` / `SH_CLIENT_SECRET_reserv`
3. `SH_CLIENT_ID_second_reserv` / `SH_CLIENT_SECRET_second_reserv`

Failover is controlled automatically by the Sentinel Hub client and is used only for quota/auth/429/stable-5xx exhaustion paths.

## Artifacts

Candidate manifest:

```text
backend/debug/runs/train_candidate_manifest.json
```

Open corpus manifest:

```text
backend/debug/runs/open_boundary_corpus_manifest.json
```

North / central / south QA summary:

```text
backend/debug/runs/release_qa_band_summary.json
```

Boundary v3 candidate outputs:

```text
backend/models/boundary_unet_v3_cpu.pth
backend/models/boundary_unet_v3_cpu.onnx
backend/models/boundary_unet_v3_cpu.norm.json
```

Yield artifacts:

```text
backend/models/yield_baseline_v2.pkl
backend/models/yield_ensemble_v2.pkl
```
