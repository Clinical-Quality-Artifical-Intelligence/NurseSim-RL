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
)


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
