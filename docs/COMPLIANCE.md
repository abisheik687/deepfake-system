# KAVACH-AI v2.0 — Compliance & Governance Framework

## 1. Regulatory Alignment

### IT Act 2000 (India)
- **Section 65B Admissibility**: Every detection generates a cryptographically signed Forensic PDF Bundle (Electronic Record) to ensure legal chain-of-custody for digital evidence.
- **Section 66D/66E**: The system is designed to detect "cheating by personation" and privacy violations while strictly handling media only for forensic validation.

### GDPR (EU)
- **Data Minimization**: KAVACH-AI processes media locally or via ephemeral ingestion. Original media is hashed and can be purged via the provided Data Retention policy.
- **Right to Explanation**: Integrated GradCAM heatmaps and Agency Agent summaries provide users with a "Reasoned Verdict," satisfying the transparent AI requirement.

## 2. Data Governance

| Data Type | Retention | Deletion Policy | Encryption |
|-----------|-----------|------------------|------------|
| Original Media | 7 Days | Automatic Purge | AES-256 (At Rest) |
| Forensic PDF | Permanent | Manual (Admin) | Signed (SHA-256) |
| User Metadata | 30 Days | On-Request | TLS 1.3 |

## 3. Ethics & Bias
KAVACH-AI utilizes a stratified FaceForensics++ training pipeline to ensure detection parity across diverse Indian demographics and skin tones, mitigating AI-induced forensic bias.
