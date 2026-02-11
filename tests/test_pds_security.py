"""
Security tests for NHS PDS Client.

Ensures that restricted records are handled securely and access is denied.
"""

import unittest
from unittest.mock import MagicMock
from nursesim_rl.pds_client import PDSClient, PDSEnvironment, RestrictedPatientError

class TestPDSSecurity(unittest.TestCase):
    """Security verification for PDS Client."""

    def setUp(self):
        self.client = PDSClient(environment=PDSEnvironment.SANDBOX)

    def test_restricted_patient_access_denied(self):
        """
        Verify that a patient with 'restricted' security label (Code R)
        raises a RestrictedPatientError and does NOT return data.
        """
        # Mock response data simulating a restricted record
        restricted_data = {
            "resourceType": "Patient",
            "id": "9000000017",
            "meta": {
                "security": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ActReason",
                        "code": "R",
                        "display": "restricted"
                    }
                ]
            },
            "name": [{"use": "official", "family": "Smythe", "given": ["Jayne"]}],
            "gender": "female",
            "birthDate": "1978-01-01"
        }

        # The parsing method should raise the error
        with self.assertRaises(RestrictedPatientError):
            self.client._parse_patient_response("9000000017", restricted_data)

if __name__ == '__main__':
    unittest.main()
