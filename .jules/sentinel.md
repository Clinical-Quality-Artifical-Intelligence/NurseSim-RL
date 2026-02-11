## 2025-05-15 - PDS Restricted Records Exposure
**Vulnerability:** The NHS PDS Client correctly parsed the 'restricted' flag (security code 'R') but did not enforce access control, returning the full patient record to the caller.
**Learning:** Parsing security metadata is not enough; enforcement logic must be explicit. Default behavior should be "deny" for flagged records unless specific authorization logic exists.
**Prevention:** In data clients, always couple the parsing of security flags with immediate enforcement checks (e.g., raising an exception) to prevent accidental data leakage.
