# VenCento — Zuliano Accent Benchmarking + Accent Infusion (Speech-to-Speech)

> **Goal:** Build an industry-grade toolkit that can **measure** and eventually **inject** the Zuliano accent into speech:
>
> 1) **Accent QA / Benchmark (Sprints 1–5):** “How Zuliano does this audio sound?” + robustness under WhatsApp/noise  
> 2) **Accent Infusion / Conversion (Sprints 6–9):** “Convert any Spanish speech → Zuliano” while preserving speaker identity and content

This repo is designed to be useful to companies shipping speech products (TTS, voice conversion, dubbing) by providing:
- an **accent embedding model** (continuous “Zuliano-ness” score),
- an **evaluation suite** (robustness + drift/regression tests),
- and a **conversion model** (accent style transfer) that can be trained and compared quantitatively.

---

## Why this matters (Industry Use-Cases)

Speech companies frequently ship features like:
- “Speak in Venezuelan Spanish” / “Speak in Zuliano style”
- “Dub Spanish but keep regional identity”
- “Voice cloning without neutralizing accent”
- “Accent sliders / tags”

**Problem:** accent quality is often evaluated via human listening. That’s slow, expensive, subjective, and doesn’t scale.

**VenCento provides automated QA**:
- **Zuliano score:** a continuous metric of accent authenticity
- **Robustness:** does the accent remain consistent under phone compression/noise?
- **Accent drift:** does your generated audio drift toward a “neutral” Spanish cluster?
- **Regression testing:** did your new model release break Zuliano?

---

## Scope & Philosophy

This project focuses on **within-Venezuela accent diversity**, starting with a practical target:
- **Binary MVP:** `Zuliano` vs `Non-Zuliano`
- Later expansion: multi-region Venezuela (Zulia, Andes, Centro/Caracas, Oriente, Llanos, Guayana, Insular)

This avoids the common pitfall “country = accent” and makes the work more novel and technically relevant.

---

## System Overview

There are two major subsystems:

1) **Accent Evaluation Stack** (Sprints 1–5)
2) **Accent Infusion Stack** (Sprints 6–9)

### 1) Accent Evaluation Stack (Benchmark + Metric)

**Input:** speech audio (2–6s recommended)  
**Outputs:**
- `p(zuliano)` (probability / calibrated confidence)
- `score_zuliano = cosine(embedding, prototype_zuliano)`
- robustness curves (score stability under compression/noise)

**Key idea:** Learn an **accent embedding** that clusters Zuliano speech together while being robust to:
- speaker identity
- text content
- channel conditions (phone/studio, WhatsApp compression)

### 2) Accent Infusion Stack (Speech-to-Speech Conversion)

**Input:** arbitrary Spanish speech audio  
**Target:** Zuliano accent style  
**Output:** same speaker identity + same words, but Zuliano-style prosody/phonetics

**Key idea:** Disentangle and recombine:
- **Content representation** (what is being said)
- **Speaker representation** (who is speaking)
- **Accent representation** (how it’s spoken — Zuliano)

Then generate a waveform conditioned on (content, speaker, target accent).

---

## Architectures (Specific)

### A) Accent Encoder / Accent Embedding Model (Sprints 3–4)

**Backbone:** Pretrained self-supervised speech model (choose one):
- HuBERT / wav2vec2 / WavLM-style encoder
- or Whisper encoder features for robust cross-domain representations

**Head:** Projection + classifier
- `z = Encoder(audio)` → frame-level features
- `h = Pool(z)` (statistics pooling or attentive pooling)
- `e = MLP(h)` → **accent embedding** (e.g., 256-d)
- `logits = Linear(e)` → Zuliano vs Non-Zuliano (or multi-region later)

**Losses (recommended multi-objective):**
1) **Classification loss**: Cross-entropy on region label
2) **Contrastive loss** (metric learning): Pull same-label embeddings closer, push different-label apart  
   - Example: supervised contrastive loss or triplet loss
3) *(Optional, advanced)* **Speaker-adversarial loss**: discourage speaker identity in `e`  
   - Add a small speaker classifier with gradient reversal
   - Goal: embeddings encode accent more than speaker timbre

**Outputs you ship:**
- `accent_encoder.pt` / `accent_encoder.onnx`
- `prototype_zuliano` (mean embedding across Zuliano training set)
- `prototype_non_zuliano` (optional)
- `score_clip()` API

---

### B) Robustness & Drift Evaluation (Sprint 5)

**Robustness tests** apply degradations at evaluation time:
- WhatsApp-like compression (Opus/low-bitrate approximation)
- additive noise (street/cafe)
- reverb
- speed perturbation (±5–10%)
- bandpass / telephone effect

**Metrics (per clip):**
- **Label stability:** % of clips whose predicted label flips under degradation
- **Score stability:** distribution of `Δ score_zuliano`
- **Calibration drift:** do probabilities remain calibrated under degradation?

**Outputs:**
- Robustness report (HTML/Markdown) with:
  - stability vs SNR curves
  - confusion matrices pre/post degradation
  - box plots of score drops
- “Golden set” evaluation: fixed set of clips to run in CI as regression tests

---

### C) Accent Infusion Model (Sprints 6–9)

#### Module 1 — Content representation (“units”)
Purpose: represent **what is being said** with minimal accent/speaker leakage.

Two viable options:

**Option 1: HuBERT Units**
- Extract frame-level SSL features
- Optionally quantize into discrete units via k-means (“speech tokens”)

**Option 2: Whisper encoder features**
- Use encoder states as content-like representations
- More robust to noisy, real-world recordings

Deliverable: `content_features.npy` per clip (or token sequences)

#### Module 2 — Speaker encoder
Purpose: preserve speaker identity/timbre.

Use a pretrained speaker embedding model (ECAPA-like embeddings or similar):
- `s = SpeakerEncoder(audio)` (fixed-dimensional, e.g., 192-d)

#### Module 3 — Accent conditioning
Use the **accent embedding** from the Accent Encoder:
- `a_zul = prototype_zuliano` (prototype-based)
or
- `a_zul = AccentEncoder(reference_zul_clip)` (reference-based)

#### Module 4 — Generator / Synthesizer
Purpose: generate waveform conditioned on (content, speaker, target accent).

Pragmatic v1 choice (fast to get working):
- **VITS-like conditional generator**
  - Inputs: content representation + speaker embedding + accent embedding (+ optional F0)
  - Outputs: waveform directly (or mel + vocoder)

Future v2 upgrade:
- flow-matching / diffusion-based decoder for higher quality, at cost of more compute/training complexity

---

## Training Strategy (Evolution)

### Stage 1 — Evaluation-first (Sprints 1–5)
You first build a reliable **accent judge** before doing conversion.

Reason: conversion can “sound different” but still not be Zuliano, or it may damage speaker identity. You need numbers.

### Stage 2 — First conversion on a narrow domain (Sprint 7)
Start with **Venezuelan→Zuliano** conversion (easier) to validate:
- does Zuliano score increase?
- does speaker similarity hold?
- does content remain intact?

### Stage 3 — Domain broadening to “Any Spanish→Zuliano” (Sprint 8)
To generalize beyond Venezuela:
- add diverse Spanish source audio (Mexico, Spain, Colombia, etc.)
- increase robustness augmentations
- prioritize content stability losses (see below)

---

## Loss Functions for Accent Infusion (Detailed)

Given source audio `x` and converted audio `y`:

### 1) Content preservation
Ensure `y` says the same thing as `x`.

Options:
- **Feature matching**: minimize distance between content representations  
  `L_content = || Content(x) - Content(y) ||`
- **ASR transcript consistency** (optional): run ASR on both, compare WER  
  (use as evaluation metric; can also be a weak loss if differentiability is not required)

### 2) Speaker preservation
Ensure `y` sounds like the same speaker.

- `L_speaker = 1 - cosine(SpeakerEnc(x), SpeakerEnc(y))`

### 3) Accent injection
Ensure `y` becomes Zuliano.

- `L_accent = - score_zuliano(y)`  
  (maximize Zuliano prototype similarity or classifier probability)

### 4) Audio reconstruction / adversarial / mel losses
Depending on generator:
- mel L1/L2, STFT loss, adversarial losses, duration/F0 constraints (optional)

---

## Evaluation Techniques (What “Good” Looks Like)

### For the Accent Encoder (Evaluation Stack)
- **Split by speaker** (non-negotiable): no speaker overlap across train/val/test
- Metrics:
  - Accuracy / F1 for Zuliano detection
  - Calibration (ECE, reliability curves)
  - Confusion patterns (when multi-region)

**Leakage tests (must run):**
- Shuffle labels at speaker-level → accuracy should collapse
- Verify channel balance: ensure Zuliano isn’t “all phone” and Non-Zuliano “all studio”

### For Robustness
- Degradation suite: compression/noise/reverb/speed
- Metrics:
  - label flip rate
  - score drop distributions (`Δ score`)
  - robustness curves vs SNR / bitrate

### For Accent Infusion
You report **three** axes (all must be tracked):

1) **Accent success**
   - Zuliano score increase: `score_zul(y) - score_zul(x)`
   - Target rank among regions (if multi-region)
2) **Speaker preservation**
   - speaker cosine similarity between x and y
3) **Content preservation / intelligibility**
   - content feature distance
   - ASR WER change (optional but compelling)

Additionally:
- “Stress test after conversion”: does the converted audio remain Zuliano under WhatsApp compression?

---

## Data Requirements (By Phase)

### Early phase (Encoder + Benchmark)
Minimum to start training reliably:
- Zuliano: **20–30 speakers**, ~2–5 minutes each
- Non-Zuliano: **40–60 speakers**, ~2–5 minutes each  
Total can be as low as **3–10 hours** for first usable results.

### Later phase (Accent Infusion)
To do conversion credibly:
- Zuliano target: ideally **10–30 hours** across many speakers and conditions
- Source Spanish: broad distribution (any Spanish) with diverse accents and recording conditions  
Labels for source accents are not required; diversity is the key.

### Required metadata fields
All clips must have:
- `speaker_id`
- `label` (at least Zuliano vs non)
- `license_or_consent` (track usage rights)
Recommended:
- `recording_type` (phone/studio)
- `state/city` (if available)
- coarse demographics (optional; avoid sensitive details)

---

## Repository Layout (Planned)

.
├── .gitattributes
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
├── todo.txt
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── eval/
├── models/
└── src/
    ├── augmentation/
    ├── preprocessing/
    └── utils/

