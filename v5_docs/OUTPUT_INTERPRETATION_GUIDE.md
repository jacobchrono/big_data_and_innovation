# Output Interpretation Guide

## Purpose

This guide explains how to interpret the synthetic claims outputs in this repository and how to use them in a fraud analytics workflow or class project discussion.

The generator produces multiple file types because large healthcare claims datasets are easier to work with when line-level data, reference tables, and summary outputs are separated.

---

## 1. Start with the summary files

Before opening the large line-level files, review the smaller CSV summaries.

Recommended reading order:

1. `dataset_summary_metrics.csv`
2. `claim_family_financial_summary.csv`
3. `fraud_breakdown.csv`
4. `rule_reason_summary.csv`
5. `provider_summary.csv`
6. `claim_status_reason_summary.csv`
7. `member_provider_history_summary.csv`

This gives you a top-down understanding of the dataset before moving into detailed records.

---

## 2. How to interpret each output

### `dataset_summary_metrics.csv`

Use this file to understand the overall scale and composition of the dataset.

Questions this file helps answer:
- How many total lines were generated?
- How many claims and encounters are represented?
- How many members and providers are included?
- What proportion of lines were labeled as fraud?
- What proportion were denied or flagged?

A low fraud rate with a higher flag rate is expected. In a realistic fraud detection workflow, systems usually over-flag relative to true fraud because the goal is to identify suspicious claims for review rather than prove misconduct automatically.

---

### `claim_family_financial_summary.csv`

Use this file to compare professional, institutional, pharmacy, member-submitted, and administrative claims.

Questions this file helps answer:
- Which claim families drive the most volume?
- Which claim families carry the highest dollars?
- Which claim families have higher denial rates or fraud rates?
- Are hospital or pharmacy claims materially different from physician claims?

This is often one of the most useful tables for a report because it shows how the dataset is not made of just one kind of claim.

---

### `fraud_breakdown.csv`

Use this file to understand what kinds of fraud were intentionally injected.

Questions this file helps answer:
- Which fraud types are most common?
- Which fraud types create the largest dollar exposure?
- Are some fraud types concentrated among fewer providers?

This file describes the known ground truth of the simulation, not what the model thinks happened.

---

### `rule_reason_summary.csv`

Use this file to understand what the rule engine is reacting to.

Questions this file helps answer:
- What rules fired most often?
- Are flags mostly driven by duplicates, excessive units, coding edits, or pricing issues?
- Which reasons produce stronger precision?

This is useful for discussing false positives. A rule may fire frequently but still have modest precision if many non-fraud lines look unusual for legitimate reasons.

---

### `provider_summary.csv`

Use this file to compare provider behavior.

Questions this file helps answer:
- Which providers generated the most lines or dollars?
- Which providers show the highest observed fraud rates?
- Which providers have unusual denial rates or flag rates?
- Are suspicious providers concentrated in specific specialties?

This file is helpful for showing that not all providers behave the same way. Most should look relatively normal, while a small subset may stand out.

---

### `claim_status_reason_summary.csv`

Use this file to understand adjudication outcomes.

Questions this file helps answer:
- How many lines were paid, denied, pended, or adjusted?
- What are the most common denial reasons?
- Do denied claims cluster in particular claim families or error types?

This file is useful when discussing operational claims workflows rather than just fraud.

---

### `member_provider_history_summary.csv`

Use this file to analyze longitudinal relationships.

Questions this file helps answer:
- Which member/provider pairs recur over time?
- Are there repeated claims from the same provider to the same member?
- Do suspicious relationships persist across many service dates?

This is especially useful for showing why time-series and relational thinking matter in fraud detection.

---

## 3. How to interpret the line-level chunks

The chunked parquet or pickle files contain the detailed line-level records.

Use them when you want to:
- train detection models
- test SQL queries
- inspect a suspicious fraud type
- build dashboards
- explore member/provider behavior over time

Because the files can be very large, do not start by opening them in Excel. Instead, use Python, DuckDB, BigQuery, or another analytical tool.

---

## 4. Recommended analysis workflow

A practical workflow for this dataset is:

### Step 1: Review summary metrics
Confirm the dataset size, fraud rate, and claim family mix.

### Step 2: Compare claim families
Look at whether institutional, pharmacy, or member-submitted claims behave differently.

### Step 3: Review known fraud injection
Use `fraud_breakdown.csv` to understand the simulation ground truth.

### Step 4: Review rule performance
Use `rule_reason_summary.csv` to understand why claims were flagged.

### Step 5: Investigate providers and episodes
Use `provider_summary.csv` and the line-level data to inspect suspicious providers or episodes.

### Step 6: Drill into line-level detail
Pull a subset of records to inspect specific fraud types or adjudication patterns.

---

## 5. Practical interpretation notes

### Fraud labels are synthetic truth
`fraud_label` tells you which records were deliberately injected as fraudulent or abusive by the simulator. It is not a real-world legal determination.

### Flags are not proof
A flagged record is not automatically fraudulent. In real workflows, flags are investigation triggers.

### Denials are not always fraud
Denied claims may reflect clerical mistakes, benefit exclusions, missing information, coding issues, or other routine payment integrity outcomes.

### Institutional and professional claims should be read together
A hospital episode may generate both facility and professional claims. Looking at only one side can hide the broader billing pattern.

### Longitudinal history matters
Repeated member/provider interactions can provide useful context that is invisible in single-line analysis.

---

## 6. File format guidance

### Parquet
Preferred format for large outputs. It is compact, columnar, and much better suited to large analytical workloads.

### Pickle (`.pkl`)
Fallback output when parquet support is unavailable. Useful in Python, but less portable across tools.

### CSV
Used for smaller summary tables and reference files because it is easy to inspect and share.

---

## 7. Suggested GitHub note

A concise note you can include in the repository README:

> The generator writes large line-level outputs in chunked parquet format when available, with pickle as a fallback. Smaller summary and reference outputs are written as CSV files. This structure makes the project easier to analyze in Python, DuckDB, or cloud warehouses such as BigQuery, while avoiding the size and performance limitations of Excel.

---

## 8. Final caveat

This dataset is designed to be realistic enough for fraud analytics prototyping and classroom discussion. It is not intended to replicate a production claims adjudication system or a fully normalized operational database.
