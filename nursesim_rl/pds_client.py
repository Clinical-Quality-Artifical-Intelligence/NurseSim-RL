"""
NHS Personal Demographics Service (PDS) FHIR API Client

A client for looking up patient demographics from the NHS PDS API
to enhance triage assessments with patient context.

Supports:
- Sandbox (no auth) for development/demo
- Integration (JWT auth) for testing
- Production (JWT auth) for live use

Reference: https://digital.nhs.uk/developer/api-catalogue/personal-demographics-service-fhir
"""

import re
import httpx
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class PDSEnvironment(Enum):
    """NHS PDS API environments."""
    SANDBOX = "sandbox"
    INTEGRATION = "int"
    PRODUCTION = "prod"


@dataclass
class PatientDemographics:
    """Normalized patient demographics from PDS."""
    nhs_number: str
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    address: Optional[str] = None
    postcode: Optional[str] = None
    gp_practice_ods: Optional[str] = None
    gp_practice_name: Optional[str] = None
    is_deceased: bool = False
    is_restricted: bool = False
    raw_response: Optional[Dict] = None


class PDSClient:
    """
    NHS Personal Demographics Service FHIR API Client.
    
    Usage:
        client = PDSClient(environment=PDSEnvironment.SANDBOX)
        patient = await client.lookup_patient("9000000009")
        print(f"Patient: {patient.full_name}, Age: {patient.age}")
    """
    
    BASE_URLS = {
        PDSEnvironment.SANDBOX: "https://sandbox.api.service.nhs.uk",
        PDSEnvironment.INTEGRATION: "https://int.api.service.nhs.uk",
        PDSEnvironment.PRODUCTION: "https://api.service.nhs.uk",
    }
    
    API_PATH = "/personal-demographics/FHIR/R4/Patient"
    
    def __init__(
        self,
        environment: PDSEnvironment = PDSEnvironment.SANDBOX,
        access_token: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Initialize PDS client.
        
        Args:
            environment: Target NHS environment (sandbox requires no auth)
            access_token: OAuth access token (required for INT/PROD)
            timeout: Request timeout in seconds
        """
        self.environment = environment
        self.access_token = access_token
        self.timeout = timeout
        self.base_url = self.BASE_URLS[environment]
        
        if environment != PDSEnvironment.SANDBOX and not access_token:
            raise ValueError(
                f"Access token required for {environment.value} environment. "
                "Use PDSEnvironment.SANDBOX for unauthenticated testing."
            )
    
    @staticmethod
    def validate_nhs_number(nhs_number: str) -> bool:
        """
        Validate NHS number using modulus 11 checksum.
        
        NHS numbers are 10 digits with the last digit being a check digit.
        
        Args:
            nhs_number: The NHS number to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Remove spaces and validate format
        nhs_number = nhs_number.replace(" ", "")
        
        if not re.match(r"^\d{10}$", nhs_number):
            return False
        
        # Modulus 11 checksum
        weights = [10, 9, 8, 7, 6, 5, 4, 3, 2]
        total = sum(int(nhs_number[i]) * weights[i] for i in range(9))
        remainder = total % 11
        check_digit = 11 - remainder
        
        if check_digit == 11:
            check_digit = 0
        elif check_digit == 10:
            return False  # Invalid NHS number
            
        return check_digit == int(nhs_number[9])
    
    def _get_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Accept": "application/fhir+json",
            "X-Request-ID": str(uuid.uuid4()),
        }
        
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
            
        return headers
    
    async def lookup_patient(self, nhs_number: str) -> PatientDemographics:
        """
        Look up patient demographics by NHS number.
        
        Args:
            nhs_number: 10-digit NHS number
            
        Returns:
            PatientDemographics object with patient details
            
        Raises:
            ValueError: If NHS number is invalid
            httpx.HTTPStatusError: If API request fails
        """
        # Clean and validate NHS number
        nhs_number = nhs_number.replace(" ", "")
        
        if not self.validate_nhs_number(nhs_number):
            raise ValueError(f"Invalid NHS number: {nhs_number}")
        
        url = f"{self.base_url}{self.API_PATH}/{nhs_number}"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()
            
        return self._parse_patient_response(nhs_number, data)
    
    def lookup_patient_sync(self, nhs_number: str) -> PatientDemographics:
        """
        Synchronous version of lookup_patient for non-async contexts.
        """
        nhs_number = nhs_number.replace(" ", "")
        
        if not self.validate_nhs_number(nhs_number):
            raise ValueError(f"Invalid NHS number: {nhs_number}")
        
        url = f"{self.base_url}{self.API_PATH}/{nhs_number}"
        
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()
            
        return self._parse_patient_response(nhs_number, data)
    
    def _parse_patient_response(
        self, nhs_number: str, data: Dict[str, Any]
    ) -> PatientDemographics:
        """Parse FHIR Patient resource into PatientDemographics."""
        
        # Extract name
        given_name = None
        family_name = None
        full_name = None
        
        names = data.get("name", [])
        if names:
            # Get official name or first available
            official_name = next(
                (n for n in names if n.get("use") == "official"),
                names[0]
            )
            given_name = " ".join(official_name.get("given", []))
            family_name = official_name.get("family", "")
            full_name = f"{given_name} {family_name}".strip()
        
        # Extract date of birth and calculate age
        date_of_birth = data.get("birthDate")
        age = self._calculate_age(date_of_birth) if date_of_birth else None
        
        # Extract gender
        gender = data.get("gender", "").capitalize()
        
        # Extract address
        address = None
        postcode = None
        addresses = data.get("address", [])
        if addresses:
            home_addr = next(
                (a for a in addresses if a.get("use") == "home"),
                addresses[0]
            )
            address_lines = home_addr.get("line", [])
            city = home_addr.get("city", "")
            postcode = home_addr.get("postalCode", "")
            address = ", ".join(filter(None, address_lines + [city, postcode]))
        
        # Extract GP practice
        gp_practice_ods = None
        gp_practice_name = None
        gps = data.get("generalPractitioner", [])
        if gps:
            gp = gps[0]
            if "identifier" in gp:
                gp_practice_ods = gp["identifier"].get("value")
            if "display" in gp:
                gp_practice_name = gp.get("display")
        
        # Check for restricted record
        is_restricted = False
        meta = data.get("meta", {})
        if "security" in meta:
            security_codes = [s.get("code") for s in meta["security"]]
            is_restricted = "R" in security_codes

        if is_restricted:
            raise ValueError("Access denied: Restricted patient record")
        
        # Check for deceased
        is_deceased = data.get("deceasedBoolean", False) or \
                      data.get("deceasedDateTime") is not None
        
        return PatientDemographics(
            nhs_number=nhs_number,
            given_name=given_name,
            family_name=family_name,
            full_name=full_name,
            date_of_birth=date_of_birth,
            gender=gender,
            age=age,
            address=address,
            postcode=postcode,
            gp_practice_ods=gp_practice_ods,
            gp_practice_name=gp_practice_name,
            is_deceased=is_deceased,
            is_restricted=is_restricted,
            raw_response=data,
        )
    
    @staticmethod
    def _calculate_age(date_of_birth: str) -> int:
        """Calculate age from date of birth string (YYYY-MM-DD)."""
        from datetime import date
        
        try:
            parts = date_of_birth.split("-")
            birth_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
            today = date.today()
            age = today.year - birth_date.year
            
            # Adjust if birthday hasn't occurred yet this year
            if (today.month, today.day) < (birth_date.month, birth_date.day):
                age -= 1
                
            return age
        except (ValueError, IndexError):
            return None


# Convenience function for quick lookups
async def lookup_patient(nhs_number: str) -> PatientDemographics:
    """
    Quick lookup using sandbox environment.
    
    For production use, instantiate PDSClient directly with credentials.
    """
    client = PDSClient(environment=PDSEnvironment.SANDBOX)
    return await client.lookup_patient(nhs_number)


def lookup_patient_sync(nhs_number: str) -> PatientDemographics:
    """Synchronous version for non-async contexts."""
    client = PDSClient(environment=PDSEnvironment.SANDBOX)
    return client.lookup_patient_sync(nhs_number)


# Test NHS numbers for sandbox
SANDBOX_TEST_PATIENTS = {
    "9000000009": "Jane Smith - Standard test patient",
    "9000000017": "Jayne Smythe - Restricted record",
    "9000000025": "John Smith - Deceased notification",
}
