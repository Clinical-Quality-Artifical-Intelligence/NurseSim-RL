import unittest
from nursesim_rl.pds_client import PDSClient, PDSEnvironment, RestrictedPatientError

class TestRestrictedPatient(unittest.TestCase):
    def test_restricted_patient_access(self):
        client = PDSClient(environment=PDSEnvironment.SANDBOX)

        # Mock response for a restricted patient
        mock_response = {
            "resourceType": "Patient",
            "id": "9000000017",
            "meta": {
                "security": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
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
            "gender": "female",
            "birthDate": "1980-01-01"
        }

        # Verify that parsing the response raises RestrictedPatientError
        with self.assertRaises(RestrictedPatientError) as cm:
            client._parse_patient_response("9000000017", mock_response)

        print(f"Correctly caught error: {cm.exception}")

if __name__ == "__main__":
    unittest.main()
