# Sentinel Journal

## 2026-02-12 - Restricted Patient Data Leakage
**Vulnerability:** The PDS Client correctly identified restricted records (flagged with 'R' in `meta.security`) but returned the full sensitive data anyway.
**Learning:** Checking a flag (`is_restricted`) is not enough; the application must enforce the restriction by raising an exception or redacting the data immediately.
**Prevention:** Implemented `RestrictedPatientError` and raised it immediately upon detecting the restricted flag in the data parsing layer (`_parse_patient_response`), preventing data from reaching the UI or API consumers.
