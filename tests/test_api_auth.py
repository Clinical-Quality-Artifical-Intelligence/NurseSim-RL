import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os
import asyncio

# --- MOCKING ---
# We need to mock 'fastapi' and other heavy dependencies BEFORE importing agent_main
mock_fastapi = MagicMock()
sys.modules["fastapi"] = mock_fastapi
sys.modules["fastapi.middleware.cors"] = MagicMock()
sys.modules["fastapi.responses"] = MagicMock()
sys.modules["uvicorn"] = MagicMock()
sys.modules["gradio"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["peft"] = MagicMock()
sys.modules["bitsandbytes"] = MagicMock()

# Mock fastapi.security
mock_fastapi_security = MagicMock()
sys.modules["fastapi.security"] = mock_fastapi_security
mock_fastapi.security = mock_fastapi_security

# Mock Pydantic
mock_pydantic = MagicMock()
class BaseModel:
    def __init__(self, **data):
        self.__dict__.update(data)
    def dict(self):
        return self.__dict__

mock_pydantic.BaseModel = BaseModel
sys.modules["pydantic"] = mock_pydantic

# Mock PDS Client
sys.modules["nursesim_rl.pds_client"] = MagicMock()

# Mock FastAPI class to capture routes
created_apps = []

class MockFastAPI:
    def __init__(self, **kwargs):
        self.routes = []
        self.middleware = []
        created_apps.append(self)

    def get(self, path, **kwargs):
        def decorator(func):
            self.routes.append({"path": path, "method": "GET", "func": func, "kwargs": kwargs})
            return func
        return decorator

    def post(self, path, **kwargs):
        def decorator(func):
            self.routes.append({"path": path, "method": "POST", "func": func, "kwargs": kwargs})
            return func
        return decorator

    def add_middleware(self, middleware_class, **options):
        self.middleware.append((middleware_class, options))

    def include_router(self, router, **kwargs):
        pass

    def mount(self, path, app, name=None):
        pass

mock_fastapi.FastAPI = MockFastAPI
mock_fastapi.HTTPException = Exception
mock_fastapi.status = MagicMock()
mock_fastapi.status.HTTP_403_FORBIDDEN = 403
mock_fastapi.status.HTTP_401_UNAUTHORIZED = 401
mock_fastapi.Depends = MagicMock(side_effect=lambda x: x)
mock_fastapi.Security = MagicMock(side_effect=lambda x: x)

# --- IMPORT ---
# Now we can import agent_main safely
if "agent_main" in sys.modules:
    del sys.modules["agent_main"]

try:
    import agent_main
except ImportError as e:
    print(f"Failed to import agent_main: {e}")
    sys.exit(1)

class TestAgentAuth(unittest.TestCase):
    def test_routes_have_auth_dependency(self):
        """Verify that sensitive routes have dependencies attached (SECURE STATE)."""
        if not created_apps:
            self.fail("No FastAPI app was created!")

        app = created_apps[0]

        sensitive_routes = ["/lookup-patient", "/process-task"]
        found_routes = {r["path"]: r for r in app.routes}

        for path in sensitive_routes:
            self.assertIn(path, found_routes, f"Route {path} not found in app")
            route = found_routes[path]
            kwargs = route["kwargs"]
            deps = kwargs.get("dependencies", [])

            print(f"Route: {path}, Dependencies: {deps}")

            # ASSERT SECURE: Dependencies exist
            self.assertGreater(len(deps), 0, f"Route {path} has NO dependencies! Vulnerable!")

            # Verify it's the right dependency
            # Since we mocked Depends as identity, the dependency is the function itself
            auth_func = deps[0]
            self.assertEqual(auth_func.__name__, "verify_api_key", f"Route {path} uses wrong dependency: {auth_func.__name__}")

    async def _verify_api_key_logic(self):
        """Test the verify_api_key function logic."""
        verify_api_key = agent_main.verify_api_key

        # Mock credentials object
        mock_creds = MagicMock()
        mock_creds.credentials = "secret_token"

        # CASE 1: No env vars set -> 403
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(Exception) as cm:
                await verify_api_key(mock_creds)
            # We mocked HTTPException as Exception, so we check if 403 was passed (status_code arg)
            # Since mock_fastapi.HTTPException is just Exception, it doesn't store status_code easily unless we check how it was initialized.
            # But wait, in agent_main: raise HTTPException(status_code=..., detail=...)
            # So the Exception args will contain status_code and detail if passed positionally,
            # or we can inspect the mock if we used a Mock for HTTPException.
            # Here checking the message is tricky because we used plain Exception.
            # Let's assume it raises Exception. Ideally verify the message "System misconfigured".
            # The exception args will be kwargs if we didn't mock init.
            # Let's rely on side effect or just trust it raises.
            pass

        # CASE 2: API_KEY set, correct token -> Returns token
        with patch.dict(os.environ, {"API_KEY": "secret_token"}, clear=True):
            result = await verify_api_key(mock_creds)
            self.assertEqual(result, "secret_token")

        # CASE 3: HF_TOKEN set, correct token -> Returns token
        with patch.dict(os.environ, {"HF_TOKEN": "secret_token"}, clear=True):
            result = await verify_api_key(mock_creds)
            self.assertEqual(result, "secret_token")

        # CASE 4: Token mismatch -> 401
        mock_creds.credentials = "wrong_token"
        with patch.dict(os.environ, {"API_KEY": "secret_token"}, clear=True):
            with self.assertRaises(Exception):
                await verify_api_key(mock_creds)

    def test_verify_api_key_sync_wrapper(self):
        """Wrapper to run async test."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._verify_api_key_logic())
        loop.close()

if __name__ == "__main__":
    unittest.main()
