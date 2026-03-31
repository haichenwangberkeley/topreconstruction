# Triplet ML Dataset Schema
schema_version: triplet-ml/v1.1

---

## 1. Purpose

This document defines the canonical schema and workflow for constructing a triplet-level machine learning dataset from analysis ntuples.

The goals are:

- Construct candidate GenJet triplets from ROOT TTrees.
- Label triplets as truth or non-truth.
- Compute physics-motivated features.
- Train a binary classifier (XGBoost BDT) to distinguish true triplets from fake ones.
- Run inference on all candidate triplets.
- Save outputs in Python-native formats (Parquet preferred).

This schema is intended for both automated coding agents and human users.

---

## 2. High-Level Pipeline Structure

The pipeline MUST be modular and separated into independent stages.

### Stage 1 — dataset_build

Input:
- ROOT ntuple(s)

Output:
- triplets_raw.parquet

Contents:
- event_id
- triplet indices (i, j, k)
- six triplet features
- truth label (is_truth)

Characteristics:
- Expensive step
- Uses streaming write
- Supports a --max-events option
- Must not depend on training configuration

---

### Stage 2 — dataset_prepare

Input:
- triplets_raw.parquet

Operations:
- Event-level train/validation/test split
- Hybrid class balancing

Output:
- train.parquet
- val.parquet
- test.parquet

Characteristics:
- Fast
- Re-runnable without rebuilding dataset

---

### Stage 3 — training

Input:
- train.parquet
- val.parquet

Output:
- trained XGBoost model
- training report

Characteristics:
- Hyperparameters configurable
- Independent of dataset_build

---

### Stage 4 — inference

Input:
- trained model
- test.parquet

Output:
- inference_test.parquet

Contents:
- event_id
- triplet indices
- features
- model score
- truth flag

All candidate triplets must be saved.

---

## 3. Input Data Requirements

### 3.1 Required Branches

GenJet branches:

- genjet_pt
- genjet_eta
- genjet_phi
- N_genjet

Jets are treated as massless. Any genjet mass branch must be ignored.

Truth triplets:

- truth_triplet_0
- truth_triplet_1
- truth_triplet_2
- truth_triplet_3

Each contains three integers indexing GenJets.

Event identifier:

The event identifier must be determined by consulting:

docs/branch_interpretation.md

An event number is expected to exist and must be used as event_id.

Fallback:
- entry index if no event number exists.

Assumption:
- No event overlap across input files.

---

## 4. Truth Triplet Definition

Truth triplets are treated as unordered sets.

A truth triplet is valid if:

0 <= index < N_genjet

for all three indices.

For each event:

truth_set = set(sorted(truth_triplet_i))

for all valid truth triplets.

A global counter must be maintained:

N_truth_triplet_total

defined as the sum of valid truth triplets over all events.

---

## 5. Candidate Triplet Construction

Candidate triplets must include all unordered combinations:

i < j < k

over all GenJets in the event.

No pt, eta, or b-tag requirements are applied.

Definitions:

- Signal: candidate triplet is in truth_set
- Background: otherwise

Matching must be set-based using sorted indices.

---

## 6. Feature Definition

Each triplet has six features.

For jets A, B, C:

### 6.1 Angular Features

- dr_ab
- dr_ac
- dr_bc

where DeltaR is computed in eta-phi space.

### 6.2 Mass Ratio Features

- mij_over_m123_ab
- mij_over_m123_ac
- mij_over_m123_bc

where:

mij is the invariant mass of a jet pair,
m123 is the invariant mass of the full triplet.

Jets are treated as massless.

Four-vector construction:

px = pt * cos(phi)
py = pt * sin(phi)
pz = pt * sinh(eta)
E  = sqrt(px^2 + py^2 + pz^2)

Invariant mass calculation:

m^2 = E^2 - |p|^2

If m^2 < 0 due to numerical precision, set m^2 = 0.

If m123 == 0, ratio features must be set to 0 or handled consistently and documented.

---

## 7. Dataset Splitting

Splitting is performed at the event level.

Default fractions:

- train: 20%
- validation: 10%
- test: 70%

Requirements:

- No event may appear in more than one split.
- All triplets inherit the split of their parent event.
- Splitting must be deterministic using a fixed seed.

---

## 8. Class Balancing

Signal triplets are significantly fewer than background triplets.

A hybrid strategy must be implemented:

1. Cap the number of background triplets per event during training.
2. Optionally apply sample weights in XGBoost.

Balancing applies only to training and validation datasets.

Test dataset must remain unmodified.

---

## 9. Storage Format

Primary format:
- Parquet

Fallback format:
- NPZ

ROOT output must not be used.

Dataset construction must support streaming writes to avoid large memory usage.

---

## 10. Reproducibility

Each pipeline stage must write:

config_snapshot.json

containing:

- schema version
- input files
- parameters
- random seed

Splitting and dataset construction must be reproducible.

---

## 11. Acceptance Checks

The implementation must verify:

- Truth triplets are treated as unordered sets.
- No event overlap exists across splits.
- All candidate triplets are generated from the full GenJet range.
- Feature values are finite and physically reasonable.
- Results are reproducible with a fixed seed.

---
