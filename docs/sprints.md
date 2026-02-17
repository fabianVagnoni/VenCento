# Project Sprints (20h each)

## Sprint 1 (20h): Define target + set up repo + data spec

### Deliverables
- One-page label spec: what counts as Zuliano, what doesn’t
- Repo skeleton (`data/`, `models/`, `eval/`, `app/`)
- Data manifest format (JSONL) with fields like: `clip_path`, `speaker_id`, `label`, `source`, `license_or_consent`

### Tasks
- Decide Zuliano definition (state/city, raised there vs born there, how to handle mixed/migrated speakers)
- Write dataset schema + consent/licensing notes (what is allowed to be used and redistributed)
- Build script scaffolding:
  - `prepare_audio.py` (resample, normalize, VAD, chunking)
  - `make_manifest.py` (build train/val/test JSONL with speaker-based splits)


## Sprint 2 (20h): Collect pilot dataset (tiny) + preprocessing pipeline

**Goal:** get to a dataset you can train on this week.

### Deliverables
- Pilot data: ~5 Zuliano speakers + ~10 non-Zuliano speakers (enough to validate end-to-end)
- VAD segmentation into 2–6 second clips
- Basic QC filtering: remove music/noise-heavy segments

### Tasks
- Implement preprocessing pipeline: resample → VAD → chunk → save clips
- Create manifests and speaker-based train/val/test split (no speaker overlap)


## Sprint 3 (20h): Train baseline “Zuliano detector” + sanity checks

### Deliverables
- Baseline classifier using pretrained encoder features + linear head
- Metrics: accuracy/F1, confusion-style breakdown, calibration plot
- Speaker leakage checks

### Tasks
- Use pretrained encoder embeddings (frozen) + train a simple linear classifier
- Ensure split is by speaker (not by clip)
- Run leakage tests:
  - Shuffle labels per speaker → accuracy should drop sharply
  - Verify no speaker appears in both train and test


## Sprint 4 (20h): Build the accent embedding model + similarity scores

### Deliverables
- Accent embedding model that outputs:
  - embedding vector (e.g., 128/256-d)
  - `p(zuliano)` (classifier output)
  - similarity score to Zuliano prototype (cosine similarity)
- Zuliano prototype embedding (mean embedding of Zuliano training clips)
- CLI command: `acentove score clip.wav`

### Tasks
- Add metric learning:
  - supervised contrastive loss or triplet loss + classification loss
- Compute prototype vectors (zuliano prototype, optional non-zuliano prototype)
- Expose scoring functions:
  - `p(zuliano)`
  - `cosine_to_zuliano_prototype`

**Note:** End of Sprint 4 = adoptable “accent metric” tool (industry-usable QA component).


## Sprint 5 (20h): Robustness suite (WhatsApp/noise) + mini dashboard

### Deliverables
- Robustness report: how stable Zuliano score is under compression/noise/speed
- Streamlit/Gradio mini app:
  - upload audio → see scores
  - optional “stress test” toggle (simulate WhatsApp/noise) and re-score

### Tasks
- Implement evaluation-time augmentations:
  - compression (WhatsApp-like), additive noise, speed perturbation (and optionally reverb)
- Report:
  - score drop distribution (delta similarity / delta `p(zuliano)`)
  - label flip rate (how often the predicted label changes)
- Build minimal UI that wraps scoring + stress tests


## Sprint 6 (20h): Content representation for conversion (the “units” step)

### Deliverables
- Pipeline that extracts content features/units from audio (stored on disk for training)
- Baseline “re-synthesis sanity test” (content → audio) if possible (even rough)

### Tasks
- Choose content representation:
  - HuBERT discrete units **OR** Whisper encoder features (pick one)
- Extract and store content features efficiently (NPY/Parquet/etc.)
- Spot-check stability across accents and recording conditions (ensure features track content)


## Sprint 7 (20h): First working conversion model (Venezuelan → Zuliano only)

**Why:** get an early “it works” before widening to all Spanish.

### Deliverables
- Rough voice conversion model that pushes speech toward Zuliano
- Evaluation showing:
  - Zuliano score increases (from Sprint 4 evaluator)
  - speaker similarity doesn’t collapse (speaker encoder cosine similarity)
  - intelligibility stays acceptable (content feature distance and/or ASR spot checks)

### Tasks
- Train a small generator conditioned on:
  - content features (from Sprint 6)
  - speaker embedding (pretrained speaker encoder)
  - target accent embedding (Zuliano prototype or reference)
- Don’t aim for perfect audio yet; aim for measurable, repeatable movement in metrics.


## Sprint 8 (20h): Expand to “any Spanish → Zuliano” (domain broadening)

### Deliverables
- Add diverse Spanish source audio (Mexico/Spain/Colombia/etc.) as inputs
- Model still increases Zuliano score while preserving content

### Tasks
- Add external Spanish audio as “source pool” (labels not required; diversity is key)
- Add domain augmentations to reduce overfitting to Venezuelan acoustics
- Retrain / fine-tune conversion model and re-run evaluation suite


## Sprint 9 (20h): Productize + “industry demo” + report

### Deliverables
- One-click demo:
  - upload → “Convert to Zuliano”
  - show before/after scores + speaker similarity + content consistency
- Clean report: metrics, failure cases, limitations, next steps
- Dockerfile + optional simple API endpoint

### Tasks
- Integrate scoring + conversion into the UI
- Build benchmark page/report with:
  - average Zuliano score increase
  - intelligibility proxy (ASR WER or content feature distance)
  - speaker similarity stats
  - robustness-after-conversion (optional but strong)
