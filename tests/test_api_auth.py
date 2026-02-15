
import os
import sys
from unittest.mock import MagicMock, patch

# Add parent directory to path to import agent_main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock heavy dependencies
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["peft"] = MagicMock()
sys.modules["bitsandbytes"] = MagicMock()
sys.modules["spaces"] = MagicMock()

# Mock Gradio
mock_gradio = MagicMock()
def mock_mount(app, *args, **kwargs):
    return app
mock_gradio.mount_gradio_app = mock_mount
sys.modules["gradio"] = mock_gradio

# Mock PDS Client
mock_pds_client = MagicMock()
mock_patient = MagicMock()
mock_patient.nhs_number = "9000000009"
mock_patient.full_name = "Jane Smith"
mock_patient.date_of_birth = "1980-01-01"
mock_patient.age = 45
mock_patient.gender = "Female"
mock_patient.address = "123 High St"
mock_patient.gp_practice_name = "Local GP"
mock_pds_client.lookup_patient_sync.return_value = mock_patient

# Set API Key for test
os.environ["API_KEY"] = "secret_key"

with patch("nursesim_rl.pds_client.PDSClient", return_value=mock_pds_client):
    import agent_main
    agent_main.agent.pds_client = mock_pds_client

    from fastapi.testclient import TestClient
    client = TestClient(agent_main.app)

    def test_security():
        print("Test 1: Unauthenticated Request (Should FAIL)")
        resp = client.post("/lookup-patient", json={"nhs_number": "9000000009"})
        print(f"Status: {resp.status_code}")
        if resp.status_code in [401, 403]:
            print("✅ PASS: Access denied as expected.")
        else:
            print(f"❌ FAIL: Access allowed! Code: {resp.status_code}")
            sys.exit(1)

        print("\nTest 2: Authenticated Request (Should PASS)")
        resp_auth = client.post(
            "/lookup-patient",
            json={"nhs_number": "9000000009"},
            headers={"Authorization": "Bearer secret_key"}
        )
        print(f"Status: {resp_auth.status_code}")
        if resp_auth.status_code == 200:
            print("✅ PASS: Access granted with valid token.")
            print(f"Data: {resp_auth.json()}")
        else:
            print(f"❌ FAIL: Valid token rejected! Code: {resp_auth.status_code} - {resp_auth.text}")
            sys.exit(1)

if __name__ == "__main__":
    test_security()
