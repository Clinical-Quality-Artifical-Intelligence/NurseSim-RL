"""
Security Verification Tests for PDS Client.

Ensures strict access controls and data protection mechanisms.
"""

import pytest
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from nursesim_rl.pds_client import PDSClient, PDSEnvironment, PatientDemographics

class TestPDSSecurity(unittest.IsolatedAsyncioTestCase):
    """
    Security verification for PDS Client.
    """

    def setUp(self):
        self.client = PDSClient(environment=PDSEnvironment.SANDBOX)

    def test_restricted_record_access_denied_sync(self):
        """
        Verify that accessing a restricted record (sync) raises a Security Exception.
        """
        # Mock restricted patient data
        mock_data = {
            "resourceType": "Patient",
            "id": "9000000017",
            "meta": {
                "security": [{"code": "R", "display": "restricted"}]
            },
            "name": [{"use": "official", "family": "Smythe", "given": ["Jayne"]}],
            "gender": "female",
            "birthDate": "1980-01-01"
        }

        with patch('httpx.Client') as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.__enter__.return_value = mock_instance

            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status.return_value = None
            mock_instance.get.return_value = mock_response

            # Should raise ValueError due to restricted access
            with self.assertRaises(ValueError) as cm:
                self.client.lookup_patient_sync("9000000017")

            self.assertIn("restricted", str(cm.exception).lower())

    async def test_restricted_record_access_denied_async(self):
        """
        Verify that accessing a restricted record (async) raises a Security Exception.
        """
        # Mock restricted patient data
        mock_data = {
            "resourceType": "Patient",
            "id": "9000000017",
            "meta": {
                "security": [{"code": "R", "display": "restricted"}]
            },
            "name": [{"use": "official", "family": "Smythe", "given": ["Jayne"]}],
            "gender": "female",
            "birthDate": "1980-01-01"
        }

        with patch('httpx.AsyncClient') as MockAsyncClient:
            mock_instance = MockAsyncClient.return_value
            mock_instance.__aenter__.return_value = mock_instance

            # Use AsyncMock for the get method
            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status.return_value = None

            # get returns a coroutine that returns the response
            mock_instance.get = AsyncMock(return_value=mock_response)

            # Should raise ValueError due to restricted access
            with self.assertRaises(ValueError) as cm:
                await self.client.lookup_patient("9000000017")

            self.assertIn("restricted", str(cm.exception).lower())

    def test_standard_record_access_allowed(self):
        """
        Verify that accessing a standard record is allowed.
        """
        # Mock standard patient data
        mock_data = {
            "resourceType": "Patient",
            "id": "9000000009",
            "name": [{"use": "official", "family": "Smith", "given": ["Jane"]}],
            "gender": "female",
            "birthDate": "1980-01-01"
        }

        with patch('httpx.Client') as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.__enter__.return_value = mock_instance

            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status.return_value = None
            mock_instance.get.return_value = mock_response

            patient = self.client.lookup_patient_sync("9000000009")
            self.assertEqual(patient.nhs_number, "9000000009")
            self.assertFalse(patient.is_restricted)

if __name__ == "__main__":
    unittest.main()
