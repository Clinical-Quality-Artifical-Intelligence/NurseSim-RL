## 2024-05-23 - Restricted Patient Data Exposure
**Vulnerability:** The PDS client was retrieving and returning full demographic details for patients flagged as "Restricted" (security code 'R') in the NHS PDS response. While the code identified the flag, it did not block access.
**Learning:** Checking for a security flag is insufficient if it doesn't interrupt the data flow. The original implementation parsed the flag but continued processing, relying on downstream consumers (which didn't exist) to handle it.
**Prevention:** Implement "fail-secure" logic at the data access layer. If a record is restricted, raise a specific exception immediately rather than returning the data object with a flag.

## 2024-05-24 - Unauthenticated API Endpoints in Prototype
**Vulnerability:** The `/lookup-patient` and `/process-task` endpoints were exposed without authentication, allowing potential access to simulated patient data and compute resources.
**Learning:** Prototype code often neglects authentication for "ease of testing", but this creates immediate debt. Verifying security controls in a dependency-constrained CI environment (missing FastAPI/Torch) required mocking the framework itself to inspect route dependencies.
**Prevention:** enforce `HTTPBearer` authentication on all endpoints by default using a reusable dependency. Use "fail-closed" logic for missing keys. Verify security controls by inspecting the application's route configuration in unit tests, even if the runtime environment is limited.

## 2025-05-27 - Timing Attack on API Key Verification
**Vulnerability:** The API key verification logic used standard string comparison (`==`) for checking `API_KEY` and `HF_TOKEN`. This is vulnerable to timing attacks, where an attacker can infer the key character by character based on the time it takes for the comparison to fail.
**Learning:** Even in high-level languages like Python, string comparison optimizations (short-circuiting) can leak information about secrets. Security-critical comparisons must always be constant-time.
**Prevention:** Use `secrets.compare_digest()` for all secret comparisons. This function is designed to run in constant time regardless of the input, preventing timing side-channel attacks.
