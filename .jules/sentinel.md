## 2025-05-15 - [CRITICAL] Unenforced Access Control on PDS Records
**Vulnerability:** The `PDSClient` library correctly identified "Restricted" (code 'R') patient records via the `is_restricted` flag but did NOT prevent the application from accessing or displaying this sensitive data.
**Learning:** Checking for a security flag is not the same as enforcing it. The client library assumed the consumer would check the flag, but `agent_main.py` and `app.py` displayed patient data regardless of this flag.
**Prevention:** Implemented a "fail-closed" mechanism within the library itself (`_parse_patient_response`) to raise a `ValueError` immediately when a restricted record is encountered, preventing downstream components from accidentally exposing it.
