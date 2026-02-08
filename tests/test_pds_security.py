
import pytest
from unittest.mock import MagicMock, patch
from nursesim_rl.pds_client import PDSClient, PDSEnvironment

# Mock response data for a restricted patient
RESTRICTED_PATIENT_RESPONSE = {
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
    "name": [{"use": "official", "given": ["Jayne"], "family": "Smythe"}],
    "gender": "female",
    "birthDate": "1990-01-01"
}

class TestPDSSecurity:
    """Security tests for PDS Client."""

    def test_restricted_patient_sync_raises_error(self):
        """Test that lookup_patient_sync raises ValueError for restricted records."""
        with patch('httpx.Client') as MockClient:
            mock_client_instance = MockClient.return_value
            mock_client_instance.__enter__.return_value = mock_client_instance

            mock_response = MagicMock()
            mock_response.json.return_value = RESTRICTED_PATIENT_RESPONSE
            mock_response.raise_for_status.return_value = None
            mock_client_instance.get.return_value = mock_response

            client = PDSClient(environment=PDSEnvironment.SANDBOX)

            # This should raise ValueError with a specific message
            with pytest.raises(ValueError, match="Access to restricted patient record is denied"):
                client.lookup_patient_sync("9000000017")
