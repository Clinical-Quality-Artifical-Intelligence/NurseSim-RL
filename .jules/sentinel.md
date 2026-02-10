## 2026-03-02 - [PDS Access Control Gap]
**Vulnerability:** Restricted patient records were accessible despite an `is_restricted` flag being parsed.
**Learning:** The `PDSClient` correctly identified restricted records but did not enforce access control by blocking them. This is a common pattern where security checks are performed but not acted upon.
**Prevention:** Ensure that security flags (like `is_restricted`) are immediately checked after parsing, and access is denied if the flag is set.
