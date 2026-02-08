# Sentinel Journal

## 2026-02-08 - Restricted PDS Record Authorization Bypass

**Vulnerability:** The PDS Client correctly identified restricted patient records (flagged with security code 'R') but did not enforce any access control, allowing restricted data (like addresses and GP details) to be returned to the application.
**Learning:** Parsing logic (`_parse_patient_response`) extracted the security flag into a boolean `is_restricted` but forgot to act on it. This is a common "check but don't enforce" pattern.
**Prevention:** Always pair security checks with immediate enforcement (e.g., raising an exception). Use unit tests that specifically target restricted/negative cases to verify enforcement.
