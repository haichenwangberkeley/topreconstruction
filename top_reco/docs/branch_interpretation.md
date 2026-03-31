# Ntuple Branch Schema and Usage Guide

This document describes the structure and intended usage of branches in the analysis ntuple.
It is written for automated analysis agents and humans. The goal is to allow an agent to
correctly interpret object collections, indexing conventions, and reconstruction logic.

All branches listed here are arrays unless otherwise stated.

This document reflects the validated interpretation obtained from dedicated diagnostic
tests comparing reconstructed triplets, truth triplets, generator-level jets, and
generator-level top quarks.

---

## 1. General Conventions

### 1.1 Object Collections

Branches are grouped by prefix. The prefix determines the physics object type and level.

| Prefix | Meaning |
|---|---|
| jet_* | Reconstructed jets (detector-level objects) |
| genjet_* | Truth-level (generator) jets |
| ph_* | Reconstructed photons |
| genph_* | Truth photons |
| ele_* | Reconstructed electrons |
| mu_* | Reconstructed muons |
| genlep_* | Truth leptons |
| top_* | Truth-level top quarks |
| bottom_* | Truth b-quarks |
| W_* | Truth W bosons |
| Wpart*_* | W decay products |
| triplet_* | Properties of generator-level jet triplets (see Section 4) |
| reco_triplet_* | ML-identified triplets of GenJets |
| truth_triplet_* | Truth-matched triplets of GenJets |

All arrays follow event-wise storage.

---

### 1.2 Four-Top Assumption and Fixed Indexing

The ntuple is designed for processes with up to four top quarks.

Therefore:

- indices run from 0 to 3
- this applies to:
  - triplet_i_*
  - reco_triplet_i
  - truth_triplet_i
  - dRmin_topi

Many events contain fewer than four tops.

In those cases:

- unused entries contain dummy placeholder values
- agents must not assume all indices correspond to physical objects
- the valid multiplicity is given by counters such as:
  - N_top
  - N_reco_triplet
  - N_gen_triplet

Agents should always use multiplicity counters when looping.

---

### 1.3 Array Interpretation Rule

If a branch is named:

object_property

then:

object_property[k]

refers to the property of the k-th object in that collection.

Example:

genjet_pt[3]

is the transverse momentum of the 4th generator-level jet.

---

## 2. Reconstructed-Level Objects

### 2.1 Jets

jet_pt  
jet_eta  
jet_phi  
jet_m  
jet_e  
jet_btag  

These describe reconstructed (detector-level) jets.

IMPORTANT:

Triplet-related variables (`reco_triplet_*`, `truth_triplet_*`, `triplet_*`)
must NOT be applied to reconstructed jets. These triplets are defined at
generator level using GenJets.

---

### 2.2 Photons

ph_pt  
ph_eta  
ph_phi  
ph_e  
ph_iso  

---

### 2.3 Electrons and Muons

Electrons:

ele_pt  
ele_eta  
ele_phi  
ele_iso  
ele_charge  

Muons:

mu_pt  
mu_eta  
mu_phi  
mu_iso  
mu_charge  

---

## 3. Truth-Level Objects

Truth-level objects represent generator information.

### 3.1 Truth Jets

genjet_pt  
genjet_eta  
genjet_phi  
genjet_m  
genjet_btag  

These are generator-level jets and form the basis of all triplet definitions.

---

### 3.2 Truth Tops

top_pt  
top_eta  
top_phi  
top_m  
top_PID  

Interpretation:

- contains up to four generator-level top quarks
- represents the true top quark kinematics from event generation
- dummy entries may be present if fewer tops exist

Agents should use N_top to determine valid entries.

---

### 3.3 Other Truth Objects

genlep_* : truth leptons  
genph_* : truth photons  
bottom_* : truth b quarks  
W_* : truth W bosons  
Wpart0_* , Wpart1_* : W decay products  

---

## 4. Triplet System (Top Reconstruction at Generator Level)

The triplet system represents identification of hadronic top candidates
using **generator-level jets (GenJets)**.

Because up to four tops are possible, four triplet slots exist.

A key point established by validation studies:

The term "reco" in `reco_triplet_*` refers to **reconstruction by a machine
learning algorithm**, not to reconstructed detector-level jets.

All triplet indices refer to GenJets.

---

### 4.1 reco_triplet_i

reco_triplet_0  
reco_triplet_1  
reco_triplet_2  
reco_triplet_3  

Each entry contains three integers.

These integers are indices into the **genjet_* collections**.

Interpretation:

- A machine learning algorithm identifies three GenJets that likely originate
  from the same top quark decay.
- The algorithm operates at truth level.
- The output is therefore a reconstructed hypothesis in generator space.

Example:

reco_triplet_0 = [1, 3, 5]

means:

genjet_pt[1], genjet_pt[3], genjet_pt[5]

belong to a GenJet triplet identified by the ML reconstruction.

IMPORTANT:

These indices must NOT be applied to jet_* branches.

---

### 4.2 truth_triplet_i

truth_triplet_0  
truth_triplet_1  
truth_triplet_2  
truth_triplet_3  

These represent truth-associated GenJet triplets.

Interpretation:

- indices into genjet_* collections
- represent GenJets that originate from the same true top quark decay
- serve as a reference for evaluating reconstruction performance

Validation results show:

- Summed four-vectors from truth_triplet_* applied to GenJets produce
  top-like kinematic distributions.
- However, the reconstructed mass distribution is broader than the
  generator-level top mass because:
  - jet clustering and radiation effects smear the invariant mass,
  - even correctly identified GenJet triplets do not perfectly reproduce
    the parent top four-vector.

This behavior is expected and physically correct.

---

### 4.3 triplet_i_*

triplet_0_pt  
triplet_0_eta  
triplet_0_phi  
triplet_0_m  

(and similarly for indices 1–3)

These represent the four-vector properties of the GenJet triplet.

Interpretation:

- derived from GenJet triplets
- correspond to the summed four-vector of the associated GenJets
- represent reconstructed top candidates at generator level

Observed behavior:

- triplet mass distributions are broader than true top mass distributions
- this reflects physical jet-level reconstruction effects rather than
  incorrect identification.

---

## 5. Event-Level Quantities

### 5.1 Object Multiplicities

N_jet  
N_jet_central  
N_ph  
N_ele  
N_mu  
N_lep  
N_genlep  
N_genjet  
N_bjet  
N_bjet_central  
N_top  
N_hadtop  
N_gen_triplet  
N_reco_triplet  

Original counts before selection:

N_jet_orig  
N_ph_orig  
N_ele_orig  
N_mu_orig  

These counters define valid array ranges for looping.

Agents should prioritize these counters over array length.

---

### 5.2 Missing Transverse Energy

Generator level:

GenMET_met  
GenMET_eta  
GenMET_phi  
GenMET_sig  

Reconstructed level:

MET_met  
MET_eta  
MET_phi  
MET_sig  

---

### 5.3 Derived Mass and Angular Observables

m_yy  
m_HT  
m_genHT  
m_tttt  
m_tt  
m_bartt  
m_ttbar1  
m_ttbar2  

Angular separations:

dR_tt  
dphi_tt  
deta_tt  
dR_anti_tt  
dphi_anti_tt  
deta_anti_tt  

---

## 6. Event Weights and Metadata

weight  
CrossSection  
Number  

Typical usage:

- weight : per-event analysis weight
- CrossSection : sample normalization
- Number : event or sample identifier

---

## 7. Agent Usage Rules (Important)

1. Never assume all four triplet or top slots are valid.
2. Always use multiplicity counters when iterating.
3. Dummy values may appear in unused entries.
4. reco_triplet_i and truth_triplet_i always index GenJets.
5. Triplet-related variables must NOT be applied to reconstructed jets.
6. triplet_* represents GenJet-level reconstruction, not true top kinematics.
7. Generator-level top variables (top_*) remain the ground truth reference.

This document serves as the canonical schema reference for automated analysis agents.
