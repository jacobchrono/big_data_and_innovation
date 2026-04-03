# MSBA 680 – Synthetic Health Claims Fraud Analytics

## Overview

This repository contains a synthetic healthcare claims data generator and fraud detection prototype developed for my **MSBA 680 Big Data and Innovation course**.

The goal of this project is to simulate a realistic claims processing environment and demonstrate how big data techniques can be applied to:

- detect fraud, waste, and abuse  
- analyze large-scale healthcare claims data  
- explore ethical and practical constraints in healthcare analytics  

Because real healthcare claims data contains **protected health information (PHI)** and is highly restricted, this project uses **synthetic data** to replicate real-world structures without exposing sensitive information.

---

## What This Project Does

This project generates a large, complex claims dataset that includes:

- **Professional claims** (physicians, specialists)
- **Institutional claims** (hospital/facility billing with revenue codes)
- **Pharmacy claims** (NDC-style data)
- **Member-submitted claims**
- **Administrative claims**

It also simulates:

- fraud patterns (duplicate billing, upcoding, excessive units, etc.)
- pricing errors and contract mismatches  
- denial logic and claim status outcomes  
- deductible and coinsurance application  
- provider behavior differences (clean vs bad actors)  
- linked **episodes of care** across facility and professional claims  

---

## Key Features

### 1. Realistic Data Generation

- Encounter-based simulation expanded into **multi-line claims**
- Facility claims generate **linked professional claims** (e.g., surgeon, radiologist, hospitalist)
- Member-provider relationships persist across time
- Multiple claim families and workflows are represented

---

### 2. Fraud Injection

The dataset includes intentionally injected fraud scenarios:

- duplicate billing  
- upcoding  
- excessive units  
- unbundling  
- provider outlier behavior  
- pricing errors and contract abuse  
- egregious charges  
- member reimbursement abuse  

Fraud is **not evenly distributed** — it is concentrated among a subset of providers, which better reflects real-world patterns.

---

### 3. Detection Methods

The prototype includes:

- rule-based detection system  
- anomaly detection (e.g., Isolation Forest)  
- labeled fraud for evaluation  

Outputs include:

- rule scores  
- primary detection reasons  
- model predictions  
- performance summaries  

---

### 4. Big Data Architecture Concepts

- data is generated in **chunks** for scalability  
- outputs are stored in **parquet (preferred)** or **pickle (fallback)**  
- summary outputs are stored as CSV  
- designed for use with:
  - Python (pandas)
  - DuckDB
  - BigQuery or other cloud warehouses  

---

## Repository Structure

```
big_data_and_innovation/
│
├── python/
│   ├── improved_synthetic_health_claims_v5.py
│
├── data/
│   ├── claims_lines_chunk_*.parquet / .pkl
│   ├── members.csv
│   ├── providers.csv
│   ├── dataset_summary_metrics.csv
│   ├── fraud_breakdown.csv
│   ├── provider_summary.csv
│   ├── rule_reason_summary.csv
│   ├── claim_family_financial_summary.csv
│   ├── member_provider_history_summary.csv
│   ├── claim_status_reason_summary.csv
│   ├── chunk_manifest.csv
│
├── DATA_DICTIONARY.md
├── OUTPUT_INTERPRETATION_GUIDE.md
├── README.md
```

---

## How to Run

Open VS Code and run in PowerShell:

```powershell
cd "C:\Users\jakeq\OneDrive\Documents\GitHub\big_data_and_innovation\python"
python .\improved_synthetic_health_claims_v5.py --n_encounters 1000000 --n_members 120000 --n_providers 6000 --chunk_size 100000
```

### Parameters

- `--n_encounters` → number of simulated encounters (before line expansion)  
- `--n_members` → number of members  
- `--n_providers` → number of providers  
- `--chunk_size` → controls memory usage  

⚠️ Note:  
1 million encounters will generate **multiple millions of claim lines**.

---

## Output Format

### Large datasets
- Stored as **Parquet** (columnar, compressed, efficient)
- Falls back to **Pickle (.pkl)** if parquet is not available

### Summary tables
- Stored as **CSV** for easy inspection

---

## Why Parquet?

Parquet is a **columnar format**, meaning data is stored by column rather than by row.

This makes it:

- faster for analytics  
- more memory efficient  
- better compressed  
- scalable for big data environments  

Excel will not be able to handle this dataset. Tools like Python, DuckDB, or BigQuery are required.

---

## Key Outputs

### dataset_summary_metrics.csv
High-level dataset overview:
- total lines  
- fraud rate  
- denial rate  
- financial averages  

---

### claim_family_financial_summary.csv
Compares:
- professional vs institutional vs pharmacy claims  
- volume and dollars by claim type  

---

### fraud_breakdown.csv
Shows:
- types of fraud injected  
- frequency and financial impact  

---

### provider_summary.csv
Highlights:
- provider behavior  
- fraud rates  
- outliers  

---

### rule_reason_summary.csv
Explains:
- why claims were flagged  
- which rules are most active  

---

### member_provider_history_summary.csv
Tracks:
- relationships across time  
- repeated interactions  
- longitudinal patterns  

---

### claim_status_reason_summary.csv
Summarizes:
- paid vs denied vs adjusted claims  
- common denial reasons  

---

## Important Concepts

### False Positives Are Expected

Any fraud detection system will generate false positives.

This project reflects that:

- fraud rate is low  
- flag rate is higher  
- flagged claims require human review  

Human judgment remains critical in real-world fraud detection workflows.

---

### Not a Production System

This is a **prototype**, not a full claims adjudication engine.

Simplifications include:

- partial deductible logic  
- simplified contract pricing  
- approximate coding edits  
- synthetic fraud labels  

However, the structure is realistic enough to demonstrate:

- big data analytics workflows  
- fraud detection concepts  
- healthcare claims complexity  

---

## Relational vs Analytical Design

This dataset is **denormalized** for analytical convenience.

In a production environment, data would likely be stored in a **relational model**, with separate tables for:

- members  
- providers  
- claims (header + lines)  
- contracts  
- payments  

This project intentionally prioritizes **ease of analysis over operational structure**.

---

## Ethics and Data Stewardship

This project avoids real claims data to:

- protect patient privacy  
- eliminate PHI risk  
- allow safe experimentation  

However, synthetic data has limitations:

- may not capture all real-world complexity  
- may introduce bias depending on assumptions  

Fraud detection systems must be used responsibly and should support—not replace—human decision-making.

---

## Final Thoughts

This project demonstrates how:

- large-scale synthetic data can support innovation  
- fraud detection can be prototyped without sensitive data  
- big data tools are essential for modern healthcare analytics  

---

## Disclaimer

This dataset is entirely synthetic and intended for educational use only.

It does not represent real patients, providers, or claims.

---

## And Finally…

This repository was built with the assistance of a highly advanced AI system that is:

- extremely knowledgeable  
- surprisingly helpful  
- absolutely, definitely, **not planning to take over the world**

Probably.
