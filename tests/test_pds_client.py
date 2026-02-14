"""
Unit tests for NHS PDS Client.

Tests NHS number validation and sandbox API connectivity.
"""

import pytest
from nursesim_rl.pds_client import (
    PDSClient,
    PDSEnvironment,
    PatientDemographics,
    lookup_patient_sync,
    SANDBOX_TEST_PATIENTS,
    RestrictedPatientError,
)
from unittest.mock import MagicMock, patch


class TestNHSNumberValidation:
    """Tests for NHS number validation."""
    
    def test_valid_nhs_number(self):
        """Test valid NHS numbers pass validation."""
        # Known valid test NHS numbers from sandbox
        valid_numbers = [
            "9000000009",
            "9000000017",
            "9000000025",
        ]
        
        for nhs_number in valid_numbers:
            assert PDSClient.validate_nhs_number(nhs_number), \
                f"Expected {nhs_number} to be valid"
    
    def test_valid_nhs_number_with_spaces(self):
        """Test NHS numbers with spaces are handled correctly."""
        assert PDSClient.validate_nhs_number("900 000 0009")
        assert PDSClient.validate_nhs_number("9000 0000 09")
    
    def test_invalid_nhs_number_wrong_length(self):
        """Test NHS numbers with wrong length fail validation."""
        assert not PDSClient.validate_nhs_number("123456789")  # 9 digits
        assert not PDSClient.validate_nhs_number("12345678901")  # 11 digits
    
    def test_invalid_nhs_number_non_numeric(self):
        """Test NHS numbers with non-numeric characters fail."""
        assert not PDSClient.validate_nhs_number("900000000A")
        assert not PDSClient.validate_nhs_number("ABCDEFGHIJ")
    
    def test_invalid_nhs_number_bad_checksum(self):
        """Test NHS numbers with invalid checksum fail."""
        # Change last digit from valid number
        assert not PDSClient.validate_nhs_number("9000000008")
        assert not PDSClient.validate_nhs_number("9000000000")


class TestPDSClientInitialization:
    """Tests for PDS client initialization."""
    
    def test_sandbox_no_auth_required(self):
        """Test sandbox environment doesn't require authentication."""
        client = PDSClient(environment=PDSEnvironment.SANDBOX)
        assert client.environment == PDSEnvironment.SANDBOX
        assert client.access_token is None
    
    def test_integration_requires_auth(self):
        """Test integration environment requires authentication."""
        with pytest.raises(ValueError, match="Access token required"):
            PDSClient(environment=PDSEnvironment.INTEGRATION)
    
    def test_production_requires_auth(self):
        """Test production environment requires authentication."""
        with pytest.raises(ValueError, match="Access token required"):
            PDSClient(environment=PDSEnvironment.PRODUCTION)
    
    def test_integration_with_token(self):
        """Test integration environment works with token."""
        client = PDSClient(
            environment=PDSEnvironment.INTEGRATION,
            access_token="test-token"
        )
        assert client.access_token == "test-token"


class TestSandboxLookup:
    """Tests for sandbox API connectivity (requires network)."""
    
    def test_lookup_jane_smith(self):
        """Test lookup of known sandbox patient."""
        client = PDSClient(environment=PDSEnvironment.SANDBOX)
        patient = client.lookup_patient_sync("9000000009")
        
        assert patient.nhs_number == "9000000009"
        assert patient.given_name is not None
        assert patient.family_name is not None
        assert patient.gender in ["Male", "Female", "Unknown", "Other"]
        assert patient.date_of_birth is not None
    
    def test_lookup_invalid_nhs_number(self):
        """Test lookup with invalid NHS number raises error."""
        client = PDSClient(environment=PDSEnvironment.SANDBOX)
        
        with pytest.raises(ValueError, match="Invalid NHS number"):
            client.lookup_patient_sync("1234567890")
    
    def test_convenience_function(self):
        """Test convenience lookup_patient_sync function."""
        patient = lookup_patient_sync("9000000009")
        
        assert isinstance(patient, PatientDemographics)
        assert patient.nhs_number == "9000000009"


class TestPatientDemographicsParsing:
    """Tests for patient response parsing."""
    
    def test_age_calculation(self):
        """Test age is calculated from DOB."""
        patient = lookup_patient_sync("9000000009")
        
        # Age should be a reasonable value
        assert patient.age is not None
        assert 0 <= patient.age <= 120
    
    def test_full_name_constructed(self):
        """Test full name is constructed from given + family."""
        patient = lookup_patient_sync("9000000009")
        
        if patient.given_name and patient.family_name:
            expected_full = f"{patient.given_name} {patient.family_name}"
            assert patient.full_name == expected_full


class TestSecurity:
    """Security tests for PDS Client."""

    def test_restricted_patient_access_denied(self):
        """Test that accessing a restricted patient raises RestrictedPatientError."""
        client = PDSClient(environment=PDSEnvironment.SANDBOX)

        # Mock response data for a restricted patient
        mock_response_data = {
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
            "birthDate": "1990-01-01"
        }

        # Patch httpx.Client.get (since lookup_patient_sync uses Client, not AsyncClient)
        # Or patch AsyncClient.get if we test lookup_patient

        # Let's test the async version since it's used in the app
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            import asyncio

            async def run_test():
                with pytest.raises(RestrictedPatientError, match="Access to patient record 9000000017 is RESTRICTED"):
                    await client.lookup_patient("9000000017")

            asyncio.run(run_test())

    def test_restricted_patient_sync_access_denied(self):
        """Test that accessing a restricted patient raises RestrictedPatientError (Sync)."""
        client = PDSClient(environment=PDSEnvironment.SANDBOX)

        # Mock response data for a restricted patient
        mock_response_data = {
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
            "name": [{"family": "Smythe", "given": ["Jayne"]}]
        }

        with patch('httpx.Client.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            with pytest.raises(RestrictedPatientError, match="Access to patient record 9000000017 is RESTRICTED"):
                client.lookup_patient_sync("9000000017")


# Integration test marker for tests requiring network
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


if __name__ == "__main__":
    # Quick manual test
    print("Testing PDS Client...")
    print("-" * 40)
    
    # Test validation
    print("NHS Number Validation:")
    print(f"  9000000009: {PDSClient.validate_nhs_number('9000000009')}")
    print(f"  1234567890: {PDSClient.validate_nhs_number('1234567890')}")
    
    # Test sandbox lookup
    print("\nSandbox Lookup:")
    try:
        patient = lookup_patient_sync("9000000009")
        print(f"  Name: {patient.full_name}")
        print(f"  DOB: {patient.date_of_birth}")
        print(f"  Age: {patient.age}")
        print(f"  Gender: {patient.gender}")
        print(f"  GP: {patient.gp_practice_name}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n✅ Tests complete!")
