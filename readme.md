
# Health Claims Synthetic Dataset Generator

## Overview
Generate a realistic health insurance claims dataset with clean data and synthetic fraud injection for fraud detection testing.

## Dataset Structure

### Claim Line Components
- **Patient Info**: ID, DOB, Gender, Member ID
- **Provider Info**: TIN, Name, Specialty, Network Status
- **Clinical Data**: Diagnosis codes (ICD-10), Procedure codes (CPT), Modifiers
- **Financial Data**: Charge amount, Payment amount, Payment type
- **Metadata**: Claim ID, Service date, Submission date, Claim status

### Billing Errors to Include
- Duplicate claims
- Incorrect modifiers
- Unbundled procedures
- Upcoding/Downcoding
- Timing issues (claims before service date)

## Generation Strategy

1. **Clean Dataset Phase**
    - Generate realistic distributions from CMS/Medicare data patterns
    - Use authentic ICD-10 and CPT code combinations
    - Realistic payment-to-charge ratios by procedure

2. **Fraud Injection Phase**
    - Flag high-risk patterns (unusual charge amounts, missing modifiers)
    - Add phantom claims and unbundling
    - Inject unusual provider-patient combinations
    - Mark fraudulent records for supervised learning

## Output Format
- CSV with claim lines
- Separate fraud label column
- Data dictionary documenting all fields

## Next Steps
- Select Python libraries (Faker, Pandas, NumPy)
- Define valid code ranges and relationships
- Create realistic payer fee schedules
