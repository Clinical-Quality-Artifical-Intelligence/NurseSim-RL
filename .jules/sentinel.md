## 2024-05-23 - Restricted Patient Data Exposure
**Vulnerability:** The PDS client was retrieving and returning full demographic details for patients flagged as "Restricted" (security code 'R') in the NHS PDS response. While the code identified the flag, it did not block access.
**Learning:** Checking for a security flag is insufficient if it doesn't interrupt the data flow. The original implementation parsed the flag but continued processing, relying on downstream consumers (which didn't exist) to handle it.
**Prevention:** Implement "fail-secure" logic at the data access layer. If a record is restricted, raise a specific exception immediately rather than returning the data object with a flag.

## 2025-05-24 - Unauthenticated PII API Endpoint
**Vulnerability:** The `/lookup-patient` endpoint was exposed without any authentication, allowing anyone to retrieve full patient demographics (PII) by NHS number.
**Learning:** Default "open" configurations for demo environments (like Hugging Face Spaces) can easily leak into production-like APIs. The assumption that `HF_TOKEN` secured the model led to overlooking API security.
**Prevention:** Always apply authentication middleware (e.g., `Depends(verify_api_key)`) to any endpoint returning PII, regardless of environment (sandbox vs. prod). Use fail-closed logic if keys are missing.
