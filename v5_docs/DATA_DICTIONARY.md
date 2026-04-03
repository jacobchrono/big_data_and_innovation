# Data Dictionary

## Overview

This repository generates a synthetic healthcare claims environment designed for fraud analytics prototyping. The outputs represent a mix of professional, institutional, pharmacy, member-submitted, and administrative claims, along with supporting reference tables and summary outputs.

The generator is intentionally denormalized at the line level for analytical convenience. In a production environment, many of these concepts would likely be stored across multiple related operational tables.

---

## Output Files

Typical output files may include:

- `claims_lines_chunk_###.parquet` or `claims_lines_chunk_###.pkl`
- `chunk_manifest.csv`
- `members.csv`
- `providers.csv`
- `dataset_summary_metrics.csv`
- `claim_family_financial_summary.csv`
- `provider_summary.csv`
- `fraud_breakdown.csv`
- `rule_reason_summary.csv`
- `member_provider_history_summary.csv`
- `claim_status_reason_summary.csv`

If parquet support is not installed in the local Python environment, chunk files may be saved as pickle files (`.pkl`) instead.

---

# 1. Line-Level Claims Data

The largest dataset is the claims line table, usually saved in chunked parquet or pickle files.

## Core identifiers

### `claim_line_id`
Unique identifier for the claim line.

### `claim_id`
Identifier for the claim. Multiple lines may belong to the same claim.

### `encounter_id`
Identifier for the originating encounter or service event.

### `episode_id`
Identifier used to link related services across an episode of care. This can tie facility and professional claims together.

### `parent_encounter_id`
When present, links a derived or related encounter back to a parent encounter.

### `member_id`
Synthetic member identifier.

### `provider_id`
Synthetic billing provider identifier.

### `servicing_provider_id`
Synthetic identifier for the servicing or rendering provider when different from the billing provider.

---

## Claim classification

### `claim_family`
High-level claim family. Typical values include:
- `professional`
- `institutional`
- `pharmacy`
- `member_submitted`
- `administrative`

### `claim_type`
More specific claim category within the broader claim family.

### `claim_source`
Indicates how the claim entered the system. This may distinguish provider-submitted, member-submitted, pharmacy, or administrative flows.

### `linked_facility_claim`
Flag showing whether the line is associated with a facility episode.

### `place_of_service`
Standard place of service style code for the service setting.

### `bill_type`
Institutional claim bill type when applicable.

### `revenue_code`
Revenue center code for institutional line items. Primarily relevant to hospital or facility claims.

---

## Dates and time fields

### `service_date`
Date of service.

### `admit_date`
Facility admit date when applicable.

### `discharge_date`
Facility discharge date when applicable.

### `submission_date`
Date the claim was submitted.

### `payment_date`
Date payment was issued.

### `first_service_date`
In summary outputs, earliest observed service date in a member/provider relationship.

### `last_service_date`
In summary outputs, latest observed service date in a member/provider relationship.

---

## Member fields

### `member_dob`
Synthetic member date of birth.

### `member_age`
Member age at or near the date of service.

### `member_gender`
Synthetic gender category.

### `plan_id`
Synthetic plan identifier.

### `product_type`
Synthetic coverage or product type.

### `network_status`
Whether the provider is in-network or out-of-network for the simulated claim context.

### `deductible_amount`
Member deductible amount for the simulated benefit design.

### `deductible_remaining`
Estimated remaining deductible at the time of claim processing.

### `applied_to_deductible`
Amount applied to deductible for the line or claim.

### `coinsurance_rate`
Member coinsurance percentage.

### `coinsurance_amount`
Member coinsurance dollar amount.

### `copay_amount`
Member copay amount when applicable.

### `member_liability_amount`
Estimated total member responsibility for the line.

---

## Provider fields

### `provider_class`
High-level provider risk or behavior grouping used in the simulation. Typical values may include:
- `clean`
- `low_risk`
- `medium_risk`
- `high_risk`

### `provider_fraud_propensity`
Simulated provider-level likelihood of fraudulent or abusive behavior.

### `provider_type`
Synthetic provider organization or practitioner type.

### `specialty`
Provider specialty. Examples may include primary care, surgery, radiology, pathology, hospitalist, anesthesiology, PT, OT, and speech therapy.

### `provider_network_status`
Synthetic provider network status.

### `contract_id`
Synthetic contract identifier used for pricing logic.

### `contract_rate`
Expected reimbursement or negotiated rate estimate used by pricing logic.

### `payment_type`
Simulated payment mechanism. Typical values may include:
- `check`
- `electronic`
- `checkless`

---

## Clinical coding fields

### `diagnosis_code`
ICD-10-CM style diagnosis code.

### `diagnosis_code_2`, `diagnosis_code_3`, etc.
Additional diagnosis fields when generated.

### `procedure_code`
CPT or HCPCS style procedure code for professional or outpatient lines.

### `modifier_1`, `modifier_2`
Procedure modifiers when applicable.

### `drg`
Synthetic DRG-style grouping field if present.

### `ndc_code`
National Drug Code style value for pharmacy claims.

### `days_supply`
Days supply for pharmacy claims.

### `quantity_dispensed`
Quantity dispensed for pharmacy claims.

### `refill_number`
Refill sequence number for pharmacy claims.

---

## Utilization and units

### `units`
Reported billing units.

### `service_line_count`
Count of service lines attached to the claim or encounter in some summary outputs.

### `length_of_stay`
Estimated inpatient stay length where applicable.

---

## Financial fields

### `billed_amount`
Submitted charge amount.

### `allowed_amount`
Amount allowed after pricing logic.

### `paid_amount`
Amount paid by the payer.

### `payer_paid_amount`
Amount paid by the payer after member liability and edits.

### `member_paid_amount`
Estimated amount paid by the member.

### `pricing_error_amount`
Amount associated with a simulated pricing error if present.

### `egregious_charge_flag`
Flag indicating unusually extreme billed charges relative to expected norms.

### `charge_to_allowed_ratio`
Ratio of billed amount to allowed amount.

### `allowed_to_contract_ratio`
Ratio of allowed amount to the expected contract amount.

---

## Adjudication and payment integrity fields

### `claim_status`
Final simulated status of the claim or line. Typical values may include:
- `paid`
- `denied`
- `pended`
- `adjusted`

### `denial_reason`
High-level denial or rejection reason when applicable.

### `pricing_method`
Pricing logic used for the line, such as contract pricing, fee schedule, percent of charge, or member reimbursement.

### `zelis_edit_flag`
Simulated external-edit flag inspired by vendor editing workflows.

### `ncci_like_edit_flag`
Flag for simplified procedure-to-procedure edit logic.

### `mue_like_edit_flag`
Flag for simplified medically unlikely edit logic.

### `duplicate_flag`
Flag for duplicate billing logic.

### `billing_error_flag`
Flag indicating a generic billing issue.

### `pricing_error_flag`
Flag indicating simulated pricing miscalculation or pricing abuse.

### `contract_mismatch_flag`
Flag indicating mismatch versus expected contract behavior.

---

## Fraud and labels

### `fraud_label`
Binary label indicating whether the line was intentionally injected as fraudulent or abusive in the simulation.

### `fraud_type`
Simulated fraud category. Typical values may include:
- `none`
- `duplicate_billing`
- `upcoding`
- `excessive_units`
- `unbundling`
- `provider_outlier`
- `pricing_error_abuse`
- `egregious_charges`
- `member_reimbursement_abuse`

### `rule_score`
Count or weighted count of triggered detection rules.

### `rule_primary_reason`
Main rule or reason associated with the flag.

### `rule_pred`
Binary output from the rule-based detection system.

### `anomaly_score`
Score from an unsupervised anomaly method when included.

### `iso_pred`
Binary anomaly prediction from Isolation Forest when included.

---

# 2. Members Reference Table (`members.csv`)

## `member_id`
Synthetic member identifier.

## `member_dob`
Synthetic date of birth.

## `member_age`
Synthetic age.

## `member_gender`
Synthetic gender category.

## `plan_id`
Synthetic plan identifier.

## `product_type`
Synthetic product type.

## `annual_deductible`
Annual deductible estimate assigned to the member.

## `coinsurance_rate`
Default member coinsurance rate.

## `chronic_score`
Synthetic chronic burden or risk indicator.

---

# 3. Providers Reference Table (`providers.csv`)

## `provider_id`
Synthetic provider identifier.

## `provider_type`
Synthetic provider type.

## `specialty`
Provider specialty.

## `provider_class`
Simulated provider risk grouping.

## `provider_fraud_propensity`
Assigned probability or relative tendency toward problematic billing patterns.

## `network_status`
Synthetic network participation status.

## `contract_id`
Synthetic contract identifier.

## `contract_rate_index`
Baseline contract pricing factor.

---

# 4. Chunk Manifest (`chunk_manifest.csv`)

This file documents the generated line-level chunk files.

## `chunk_number`
Sequential chunk index.

## `file_name`
Name of the generated parquet or pickle file.

## `format`
Storage format, typically `parquet` or `pkl`.

## `row_count`
Rows written in the chunk.

## `min_service_date`
Earliest service date in the chunk.

## `max_service_date`
Latest service date in the chunk.

---

# 5. Dataset Summary Metrics (`dataset_summary_metrics.csv`)

High-level metrics describing the generated dataset.

## Common fields
- `metric_name`
- `metric_value`

## Example metrics
- total lines
- total claims
- total encounters
- unique members
- unique providers
- fraud rate
- denial rate
- paid rate
- average billed amount
- average allowed amount
- average paid amount

---

# 6. Claim Family Financial Summary (`claim_family_financial_summary.csv`)

Summarizes volume and dollars by claim family.

## Typical fields
- `claim_family`
- `line_count`
- `claim_count`
- `billed_amount_sum`
- `allowed_amount_sum`
- `paid_amount_sum`
- `avg_billed_amount`
- `avg_allowed_amount`
- `avg_paid_amount`
- `fraud_rate`
- `denial_rate`

---

# 7. Provider Summary (`provider_summary.csv`)

Summarizes billing and fraud signals by provider.

## Typical fields
- `provider_id`
- `provider_type`
- `specialty`
- `provider_class`
- `line_count`
- `claim_count`
- `member_count`
- `avg_billed_amount`
- `avg_allowed_amount`
- `avg_paid_amount`
- `fraud_rate`
- `flag_rate`
- `denial_rate`
- `first_service_date`
- `last_service_date`

---

# 8. Fraud Breakdown (`fraud_breakdown.csv`)

Summarizes known injected fraud by type.

## Typical fields
- `fraud_type`
- `line_count`
- `claim_count`
- `provider_count`
- `member_count`
- `avg_billed_amount`
- `avg_allowed_amount`
- `avg_paid_amount`

---

# 9. Rule Reason Summary (`rule_reason_summary.csv`)

Explains why the rule-based system generated flags.

## Typical fields
- `rule_primary_reason`
- `flag_count`
- `share_of_all_flags`
- `true_fraud_count`
- `precision_within_reason`

---

# 10. Member/Provider History Summary (`member_provider_history_summary.csv`)

Describes repeated interaction history across time.

## Typical fields
- `member_id`
- `provider_id`
- `first_service_date`
- `last_service_date`
- `line_count`
- `claim_count`
- `total_paid_amount`
- `avg_paid_amount`
- `distinct_fraud_types`

This file is useful for identifying longitudinal relationships and suspicious repeat billing patterns.

---

# 11. Claim Status Reason Summary (`claim_status_reason_summary.csv`)

Summarizes adjudication outcomes and denial reasons.

## Typical fields
- `claim_status`
- `denial_reason`
- `line_count`
- `share_of_lines`
- `billed_amount_sum`
- `allowed_amount_sum`
- `paid_amount_sum`

---

## Notes and Caveats

1. All data are synthetic and created for educational or prototyping purposes.
2. Code combinations are intended to be plausible, not production-valid in every edge case.
3. Some fields may appear only for certain claim families.
4. Some outputs may differ slightly by script version as the simulation evolves.
5. The dataset is analytically useful but not a substitute for a production claims data model.
