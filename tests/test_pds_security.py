"""
Security tests for NHS PDS Client.

Ensures restricted records are not accessible.
"""

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from nursesim_rl.pds_client import PDSClient, PDSEnvironment

class TestPDSSecurity(unittest.IsolatedAsyncioTestCase):
    def test_restricted_patient_access_sync(self):
        """Test that accessing a restricted patient synchronously raises ValueError."""
        # Mock response for a restricted patient
        mock_response = {
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
            "birthDate": "1980-01-01",
            "gender": "female"
        }

        client = PDSClient(environment=PDSEnvironment.SANDBOX)

        # Patch httpx.Client to return our mock response
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.return_value.status_code = 200
            mock_client.get.return_value.json.return_value = mock_response
            mock_client_cls.return_value.__enter__.return_value = mock_client

            # This should raise ValueError due to restricted record
            with self.assertRaises(ValueError) as cm:
                client.lookup_patient_sync("9000000017")

            # We expect a specific error message about restricted access
            self.assertIn("Restricted patient record", str(cm.exception))

    async def test_restricted_patient_access_async(self):
        """Test that accessing a restricted patient asynchronously raises ValueError."""
        # Mock response for a restricted patient
        mock_response = {
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
            "birthDate": "1980-01-01",
            "gender": "female"
        }

        client = PDSClient(environment=PDSEnvironment.SANDBOX)

        # Patch httpx.AsyncClient to return our mock response
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()

            mock_response_obj = MagicMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response

            mock_client.get.return_value = mock_response_obj
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            # This should raise ValueError due to restricted record
            with self.assertRaises(ValueError) as cm:
                await client.lookup_patient("9000000017")

            self.assertIn("Restricted patient record", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
