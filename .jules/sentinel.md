## 2024-05-23 - Restricted Patient Data Exposure
**Vulnerability:** The PDS client was retrieving and returning full demographic details for patients flagged as "Restricted" (security code 'R') in the NHS PDS response. While the code identified the flag, it did not block access.
**Learning:** Checking for a security flag is insufficient if it doesn't interrupt the data flow. The original implementation parsed the flag but continued processing, relying on downstream consumers (which didn't exist) to handle it.
**Prevention:** Implement "fail-secure" logic at the data access layer. If a record is restricted, raise a specific exception immediately rather than returning the data object with a flag.

## 2024-05-24 - Unauthenticated API Endpoints in Prototype
**Vulnerability:** The `/lookup-patient` and `/process-task` endpoints were exposed without authentication, allowing potential access to simulated patient data and compute resources.
**Learning:** Prototype code often neglects authentication for "ease of testing", but this creates immediate debt. Verifying security controls in a dependency-constrained CI environment (missing FastAPI/Torch) required mocking the framework itself to inspect route dependencies.
**Prevention:** enforce `HTTPBearer` authentication on all endpoints by default using a reusable dependency. Use "fail-closed" logic for missing keys. Verify security controls by inspecting the application's route configuration in unit tests, even if the runtime environment is limited.

## 2024-05-25 - Unauthenticated Gradio UI in Hybrid Agent
**Vulnerability:** The `agent_main.py` application mounted a Gradio UI at the root (`/`) without authentication, bypassing the strict API key validation enforced on other endpoints (`/process-task`). This exposed the PDS sandbox and LLM capabilities to unauthenticated users.
**Learning:** Hybrid applications that mix API routes (FastAPI) and UI mounts (Gradio) often apply security controls inconsistently. API dependencies (like `HTTPBearer`) do not automatically protect mounted sub-applications unless explicitly configured.
**Prevention:** Always verify authentication for ALL entry points. When mounting Gradio apps, explicitly configure the `auth` parameter or wrap the mount in a middleware that enforces authentication.
## 2024-05-24 - Timing Attack in API Key Verification
**Vulnerability:** The `verify_api_key` function used string equality (`==`) to compare the provided token with the stored secrets. This allows an attacker to deduce the key character by character by measuring the response time (CWE-208).
**Learning:** Standard string comparison operators in Python (and most languages) are not constant-time and terminate early on mismatch. Security-sensitive comparisons must use constant-time algorithms.
**Prevention:** Use `secrets.compare_digest` for all secret comparisons to prevent timing attacks.
## 2025-05-27 - Timing Attack on API Key Verification
**Vulnerability:** The API key verification logic used standard string comparison (`==`) for checking `API_KEY` and `HF_TOKEN`. This is vulnerable to timing attacks, where an attacker can infer the key character by character based on the time it takes for the comparison to fail.
**Learning:** Even in high-level languages like Python, string comparison optimizations (short-circuiting) can leak information about secrets. Security-critical comparisons must always be constant-time.
**Prevention:** Use `secrets.compare_digest()` for all secret comparisons. This function is designed to run in constant time regardless of the input, preventing timing side-channel attacks.

## 2025-05-28 - Unauthenticated Gradio UI Exposure
**Vulnerability:** The Gradio user interfaces mounted in `agent_main.py` and `app.py` were exposed without authentication, bypassing the API security controls and allowing public access to PDS lookup and model inference.
**Learning:** Securing API endpoints (FastAPI dependencies) does not automatically secure mounted sub-applications or UIs. Framework integration points (like `gr.mount_gradio_app`) often have their own separate security configurations that must be explicitly enabled.
**Prevention:** Explicitly configure authentication for all exposed interfaces. Implement a "Fail-Secure" default: if no authentication keys are provided in the environment, the application should lock itself down (e.g., by generating a random unreachable password) rather than defaulting to open access.
