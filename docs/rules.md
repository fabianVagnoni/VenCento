# VenCento — Agent Rules & Project Guidelines

> **Purpose:** This document is the authoritative reference for any agent or engineer working on this codebase. Read it before writing any code.

---

## 1. General Objectives

**VenCento** is an industry-grade speech toolkit for measuring and injecting the **Zuliano accent** (Zulia state, Venezuela) into Spanish speech. It provides:

- **Accent QA / Benchmark (Sprints 1–5):** A continuous "Zuliano-ness" score and robustness evaluation under real-world degradation (WhatsApp compression, noise, reverb).
- **Accent Infusion / Conversion (Sprints 6–9):** A speech-to-speech conversion system that transforms any Spanish speech to Zuliano while preserving speaker identity and linguistic content.

### Two Major Subsystems

| Subsystem | Sprints | Output |
|-----------|---------|--------|
| **Accent Evaluation Stack** | 1–5 | `accent_encoder.pt`, Zuliano prototype, `score_zuliano()` API, robustness report |
| **Accent Infusion Stack** | 6–9 | Voice conversion model, one-click demo, Docker API |

### Current Project Focus (Sprints 1–5)

The immediate goal is to build a reliable **accent judge** before building any conversion model. The evaluation stack must be solid — it will serve as the ground truth for all later conversion experiments.

Key deliverables per sprint:
- **Sprint 1:** Label spec, repo skeleton, JSONL manifest schema
- **Sprint 2:** Pilot dataset (~5 Zuliano + ~10 non-Zuliano speakers), VAD preprocessing pipeline
- **Sprint 3:** Baseline Zuliano detector (frozen SSL encoder + linear head), speaker-leakage tests
- **Sprint 4:** Full accent embedding model with metric learning, prototype vectors, CLI scoring
- **Sprint 5:** Robustness suite (WhatsApp/noise/speed), Streamlit/Gradio mini demo

### Target Metric Axes

Any model or experiment must be reported on these three axes (for infusion; two for evaluation):

1. **Accent authenticity** — `score_zuliano = cosine(embedding, prototype_zuliano)` and `p(zuliano)`
2. **Speaker preservation** — cosine similarity of speaker embeddings between source and converted audio
3. **Content preservation** — content feature distance and/or ASR WER change

---

## 2. Best Practices

### 2.1 Data

- **Speaker-based splits are non-negotiable.** A speaker must never appear in more than one of train/val/test. Use `src.data.splits.split_manifest()` which enforces this by design.
- **Run speaker-leakage tests** before reporting any accuracy number. Shuffle labels at the speaker level; accuracy must collapse. If it doesn't, the split is leaking.
- **Channel balance.** Verify that Zuliano is not systematically "all phone" and non-Zuliano "all studio". An imbalanced channel distribution will inflate classifier performance.
- **Manifest schema** (defined in `src.data.manifest`): every chunk record must carry `audio_id`, `chunk_id`, `zuliano` (bool), `speaker_id`, and `chunk_path`. Optional: `sr`, `duration`, `accent`.
- **Label mapping** must be derived once from the training set (`build_label_to_id`) and reused for val/test. Never infer the mapping independently per split.
- **Pilot-first.** Validate the end-to-end pipeline on a tiny dataset (~5 Zuliano speakers, ~10 non-Zuliano) before scaling. A broken pipeline found at scale is expensive.

### 2.2 Preprocessing

- All audio is resampled to **16 000 Hz mono** (`DEFAULT_SR`), stored as **s16le PCM** flat files.
- The preprocessing pipeline is: `canonize → load → RMS-normalize (–14 dBFS) → Silero VAD → chunk (2–10 s) → save PCM`.
- VAD segments are the atomic unit. Each segment becomes one manifest record.
- Normalization target is **–14 dBFS** (broadcast loudness standard) to reduce gain variance across speakers and recording conditions.
- Canonical PCM files are stored under `data/raw/canonical/`; chunks under `data/processed/chunks/`.

### 2.3 Augmentation

- **Training only.** `OnTheFlyAugmenter` must only be passed to the training dataset. Val and test datasets receive no augmenter.
- The augmenter is an `nn.Module` (picklable) so DataLoader workers can use it safely.
- The augmentation order is: `random_crop → gain → speed_perturb → codec_opus → final_crop`. Length-changing operations (speed, codec) happen before the final crop.
- **Opus codec simulation** (`p=0.15`) is critical for robustness to WhatsApp-style transmission.
- Config lives in `src/augmentation/config.yaml` and is loaded via `load_augment_config()`. Do not hardcode augmentation parameters in Python.
- `CollateFn.target_samples` must be aligned with `augmenter.cfg.crop_len`. Misalignment causes silent shape errors.

### 2.4 Model Architecture

- **Backbone:** Pretrained self-supervised speech model from HuggingFace — `wav2vec2`, `HuBERT`, or `WavLM`. These are not trained from scratch. Use frozen or lightly fine-tuned encoders.
- **Head:** Statistics pooling (or attentive pooling) → MLP → accent embedding (128–256-d) → linear classifier.
- **Loss (multi-objective):**
  1. Classification loss (cross-entropy)
  2. Metric learning loss (supervised contrastive or triplet) to structure the embedding space
  3. *(Optional, advanced)* Speaker-adversarial loss via gradient reversal to decouple accent from speaker timbre
- **Attention mask:** Always pass `attention_mask` from `CollateFn.Batch` to the HuggingFace model. This matches the `Wav2Vec2ForSequenceClassification` / `HubertForSequenceClassification` API convention.

### 2.5 Evaluation

- Calibration matters. Report ECE and reliability curves alongside accuracy/F1.
- Robustness metrics: label flip rate, score drop distribution (`Δ score_zuliano`), stability vs SNR curves.
- For infusion evaluation, all three axes (accent, speaker, content) must be tracked together — a model that gains Zuliano score by destroying content is not acceptable.
- "Stress test after conversion": run converted audio through the robustness suite. A conversion that collapses under WhatsApp compression is not production-ready.

### 2.6 Code Quality

- Every public function and class must have a **docstring** that states: purpose, parameters (with types), return value, and any exceptions raised.
- Module-level docstrings must include an **integration map** listing which other modules this module depends on and a **public API** section listing the exported symbols.
- Use `TypedDict` / `dataclass(frozen=True)` for all structured data contracts (manifests, configs, batches). This makes interfaces explicit and type-checkable.
- Constants that mirror config values (e.g. `DEFAULT_SR = 16_000`) must be defined once and imported everywhere. Never repeat magic numbers.
- Tests use `pytest` with `tmp_path` fixtures. Tests must not touch `data/` on disk. All I/O in tests must use temporary paths.
- Use `from __future__ import annotations` for forward-compatible type hints.

### 2.7 Experiment Tracking

- Every training run must log: model config, data split stats (n_chunks, n_speakers per split), augmentation config, and all evaluation metrics.
- Prototype vectors (`prototype_zuliano`, optional `prototype_non_zuliano`) must be saved alongside the model checkpoint so scoring is always self-contained.
- Label-to-id mappings must be saved as JSON next to every checkpoint (`save_label_to_id`).

---

## 3. Development Philosophy

### 3.1 Parsimonious by Default

**Write the smallest amount of code that correctly solves the problem.** Prefer:
- A 30-line function with a clear docstring over a 150-line class hierarchy.
- A single well-named constant over a configuration subsystem that isn't needed yet.
- Composition of existing utilities over new abstractions.

Before adding any new abstraction, ask: *"Does this already exist in the codebase or in a well-maintained library?"*

### 3.2 Don't Reinvent the Wheel — Leverage Existing Resources First

Before implementing anything, **identify what is already available in the ecosystem**:

| Need | What to use — do NOT reimplement |
|------|----------------------------------|
| SSL speech backbone (HuBERT, wav2vec2, WavLM) | [`transformers`](https://huggingface.co/docs/transformers) — `AutoModel.from_pretrained` |
| Speaker embeddings | [`speechbrain`](https://github.com/speechbrain/speechbrain) ECAPA-TDNN, or `pyannote.audio` |
| Supervised contrastive loss | [`pytorch-metric-learning`](https://github.com/KevinMusgrave/pytorch-metric-learning) |
| Speech augmentation (noise, RIR, codec) | [`torch-audiomentations`](https://github.com/asteroid-team/torch-audiomentations) or `audiomentations` |
| VAD | [`silero-vad`](https://github.com/snakers4/silero-vad) — already in `requirements.txt` |
| Voice activity / diarisation | `pyannote.audio` |
| Whisper features | `openai-whisper` or HuggingFace `openai/whisper-*` |
| HuBERT discrete units / k-means | [`fairseq`](https://github.com/facebookresearch/fairseq) or HuggingFace speech models |
| Calibration metrics (ECE) | [`netcal`](https://github.com/EFS-OpenSource/calibration-framework) or `torchmetrics` |
| VITS-style TTS/VC generator | [`Coqui TTS`](https://github.com/coqui-ai/TTS) or [`VITS`](https://github.com/jaywalnut310/vits) original repo |
| Evaluation UI | `gradio` or `streamlit` |
| Metrics (F1, AUC) | `torchmetrics` or `sklearn.metrics` |

**The decision process before any implementation:**
1. Search HuggingFace Hub, GitHub, and PyPI for an existing solution.
2. If one exists: integrate it. Write a thin wrapper if the interface doesn't match.
3. If no suitable solution exists: implement minimally, document why no existing solution was used.

### 3.3 Evaluation-First Development

Build the **accent judge before the conversion model**. A conversion model without a reliable automatic metric is unverifiable and untrustworthy. This is the fundamental design constraint driving the sprint ordering.

### 3.4 Staged Complexity

Follow the sprint progression:
1. **Frozen backbone + linear head** (Sprint 3) before fine-tuning.
2. **Metric learning** (Sprint 4) before adversarial losses.
3. **Venezuelan→Zuliano conversion** (Sprint 7) before "any Spanish→Zuliano" (Sprint 8).
4. **VITS-style generator** before flow-matching/diffusion.

Do not skip stages. Early stages produce the baselines and sanity checks that make later stages trustworthy.

### 3.5 Explicit Contracts Over Implicit Conventions

- **JSONL manifests** are the canonical data contract between preprocessing and training. The manifest schema is defined as a `TypedDict` in `src.data.manifest` — treat it as an interface specification.
- **PCM s16le** is the canonical audio storage format. Do not introduce new formats without updating the schema and all downstream readers.
- **`CollateFn.Batch`** is a frozen dataclass — it is the contract between the DataLoader and the model. Do not pass raw dicts to the model.

### 3.6 Configuration Over Hardcoding

All pipeline parameters (sample rate, chunk lengths, augmentation probabilities, model hyperparameters) live in YAML config files, not in Python source. Python constants (e.g. `DEFAULT_SR`) exist only to expose config values to typed code; they must mirror the YAML, not replace it.

### 3.7 Documentation Standards

- **Module docstring:** purpose, integration map (dependencies), public API summary.
- **Class docstring:** purpose, parameter table, usage example.
- **Function docstring:** one-line summary, `Parameters` / `Returns` / `Raises` sections (NumPy style).
- Comments in code body: only for non-obvious decisions. Never narrate what the code does.

---

## 4. Repository Layout

```
.
├── data/
│   ├── raw/             # original audio + raw manifest.jsonl + canonical PCMs
│   └── processed/       # chunks (PCM), processed manifest.jsonl, splits/
├── docs/                # project documentation (sprints.md, rules.md, ...)
├── eval/                # evaluation scripts, robustness reports
├── models/              # saved checkpoints, prototype vectors, label_to_id.json
├── scripts/             # one-off data import scripts (cv_importer.py, ...)
├── src/
│   ├── augmentation/    # OnTheFlyAugmenter, codec_opus, basic_operations, config.yaml
│   ├── data/            # manifest schema, splits, dataset, collate
│   ├── preprocessing/   # canonize, VAD, normalize, preprocessing pipeline, config.yaml
│   └── utils/           # io_helpers, jsonl_helpers, audio_helpers, general_utils
└── tests/               # pytest unit tests (no disk I/O outside tmp_path)
```

### Module Dependency Direction

```
scripts/ → src/data/, src/utils/
src/data/ → src/utils/, src/augmentation/
src/preprocessing/ → src/utils/
src/augmentation/ → src/utils/
tests/ → src/**
```

No circular dependencies. `src/utils/` has no intra-`src` imports.
