import unittest
from unittest.mock import MagicMock, patch
from nursesim_rl.pds_client import PDSClient, PDSEnvironment

class TestPDSSecurity(unittest.TestCase):
    def setUp(self):
        self.client = PDSClient(environment=PDSEnvironment.SANDBOX)

    @patch('httpx.Client')
    def test_restricted_record_access(self, mock_client_cls):
        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "resourceType": "Patient",
            "id": "9000000017",
            "meta": {
                "security": [
                    {
                        "code": "R",
                        "display": "restricted"
                    }
                ]
            },
            "name": [
                {
                    "use": "official",
                    "family": "Smythe",
                    "given": ["Jayne"]
                }
            ],
            "birthDate": "1980-01-01",
            "gender": "female"
        }
        mock_response.status_code = 200

        # Setup the mock client
        mock_client_instance = mock_client_cls.return_value
        mock_client_instance.__enter__.return_value.get.return_value = mock_response

        # This should now fail with ValueError
        with self.assertRaises(ValueError) as cm:
            self.client.lookup_patient_sync("9000000017")

        self.assertIn("Access to restricted patient record denied", str(cm.exception))
        print("Test passed: Restricted patient record access denied (SECURITY FIX VERIFIED)")

if __name__ == '__main__':
    unittest.main()
