
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add current directory to path
sys.path.append(os.getcwd())

# Mock external dependencies
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["peft"] = MagicMock()
sys.modules["bitsandbytes"] = MagicMock()
sys.modules["uvicorn"] = MagicMock()
sys.modules["gradio"] = MagicMock()
sys.modules["httpx"] = MagicMock()

# Mock Pydantic
pydantic_mock = MagicMock()
pydantic_mock.BaseModel = object
sys.modules["pydantic"] = pydantic_mock

# Mock FastAPI
fastapi_mock = MagicMock()
sys.modules["fastapi"] = fastapi_mock
sys.modules["fastapi.security"] = MagicMock()
sys.modules["fastapi.responses"] = MagicMock()
sys.modules["fastapi.middleware.cors"] = MagicMock()

# Mock PDS Client
pds_module = MagicMock()
sys.modules["nursesim_rl.pds_client"] = pds_module
pds_module.PDSClient = MagicMock()
pds_module.PDSEnvironment = MagicMock()
pds_module.PatientDemographics = MagicMock()
pds_module.RestrictedPatientError = ValueError

# Ensure agent_main uses the mocked gradio
gradio_mock = sys.modules["gradio"]
gradio_mock.mount_gradio_app = MagicMock()

class TestGradioAuth(unittest.TestCase):
    def setUp(self):
        # Reset mock
        gradio_mock.mount_gradio_app.reset_mock()

        # Clean up modules
        if "agent_main" in sys.modules:
            del sys.modules["agent_main"]

        # Save original environ
        self._original_environ = os.environ.copy()

    def tearDown(self):
        # Restore environ
        os.environ.clear()
        os.environ.update(self._original_environ)

    def test_gradio_mount_auth_with_api_key(self):
        os.environ["API_KEY"] = "secret_api_key"
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]

        import agent_main

        calls = gradio_mock.mount_gradio_app.call_args_list
        self.assertTrue(len(calls) > 0, "mount_gradio_app should be called")
        _, kwargs = calls[0]
        self.assertEqual(kwargs.get("auth"), ("admin", "secret_api_key"))

    def test_gradio_mount_auth_with_hf_token(self):
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]
        os.environ["HF_TOKEN"] = "hf_token_value"

        import agent_main

        calls = gradio_mock.mount_gradio_app.call_args_list
        self.assertTrue(len(calls) > 0, "mount_gradio_app should be called")
        _, kwargs = calls[0]
        self.assertEqual(kwargs.get("auth"), ("admin", "hf_token_value"))

    def test_gradio_mount_no_auth_warning(self):
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]

        import agent_main

        calls = gradio_mock.mount_gradio_app.call_args_list
        self.assertTrue(len(calls) > 0, "mount_gradio_app should be called")
        _, kwargs = calls[0]
        self.assertIsNone(kwargs.get("auth"))

if __name__ == "__main__":
    unittest.main()
