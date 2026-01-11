"""
Patient Generator for NurseSim-RL

Generates synthetic patient scenarios based on Manchester Triage System categories.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Patient:
    """Represents a patient presenting to A&E."""
    id: str
    chief_complaint: str
    vitals: Dict[str, float]
    history: str
    true_category: int  # 1-5 (Ground truth for reward calculation)
    time_arrived: int
    

# Manchester Triage System Scenarios
SCENARIOS = {
    # Category 1: Immediate (Red) - Life threatening
    1: [
        {
            "chief_complaint": "I can't breathe... my chest is crushing... the pain goes down my arm.",
            "vitals": {"hr": 120, "bp_sys": 85, "bp_dia": 50, "spo2": 88, "rr": 32, "temp": 36.5, "avpu": "V"},
            "history": "65yo male, known cardiac history, sudden onset 20 mins ago."
        },
        {
            "chief_complaint": "He collapsed and isn't responding to me!",
            "vitals": {"hr": 0, "bp_sys": 0, "bp_dia": 0, "spo2": 0, "rr": 0, "temp": 35.0, "avpu": "U"},
            "history": "72yo male found unresponsive by wife. Bystander CPR in progress."
        },
        {
            "chief_complaint": "My face is swelling up and I can't swallow... I ate shellfish.",
            "vitals": {"hr": 130, "bp_sys": 70, "bp_dia": 40, "spo2": 85, "rr": 28, "temp": 37.0, "avpu": "A"},
            "history": "28yo female, known shellfish allergy, stridor audible."
        },
    ],
    
    # Category 2: Very Urgent (Orange) - Time critical
    2: [
        {
            "chief_complaint": "I have the worst headache of my life. It came on suddenly.",
            "vitals": {"hr": 90, "bp_sys": 180, "bp_dia": 100, "spo2": 97, "rr": 18, "temp": 37.2, "avpu": "A"},
            "history": "45yo female, sudden onset occipital headache, photophobia, neck stiffness."
        },
        {
            "chief_complaint": "My little boy is having a fit and won't stop!",
            "vitals": {"hr": 150, "bp_sys": 90, "bp_dia": 55, "spo2": 90, "rr": 24, "temp": 39.5, "avpu": "U"},
            "history": "3yo male, febrile seizure ongoing for 8 minutes."
        },
        {
            "chief_complaint": "I fell and I can't feel my legs.",
            "vitals": {"hr": 100, "bp_sys": 140, "bp_dia": 85, "spo2": 98, "rr": 20, "temp": 36.8, "avpu": "A"},
            "history": "55yo male, fell from ladder, complaining of neck pain, no sensation below T4."
        },
    ],
    
    # Category 3: Urgent (Yellow)
    3: [
        {
            "chief_complaint": "I've had abdominal pain for 2 days. It's getting worse and I'm vomiting.",
            "vitals": {"hr": 105, "bp_sys": 110, "bp_dia": 70, "spo2": 97, "rr": 20, "temp": 38.2, "avpu": "A"},
            "history": "32yo female, RIF pain, guarding, rebound tenderness."
        },
        {
            "chief_complaint": "I've been short of breath for a few days. It's worse when I walk.",
            "vitals": {"hr": 95, "bp_sys": 125, "bp_dia": 80, "spo2": 92, "rr": 24, "temp": 37.0, "avpu": "A"},
            "history": "70yo male, COPD, productive cough, increased work of breathing."
        },
        {
            "chief_complaint": "I cut my hand on a knife. It won't stop bleeding.",
            "vitals": {"hr": 88, "bp_sys": 130, "bp_dia": 82, "spo2": 99, "rr": 16, "temp": 36.9, "avpu": "A"},
            "history": "40yo male, deep laceration to palm, tendon visible, bleeding controlled with pressure."
        },
    ],
    
    # Category 4: Standard (Green)
    4: [
        {
            "chief_complaint": "I've had a sore throat and cough for 3 days.",
            "vitals": {"hr": 78, "bp_sys": 120, "bp_dia": 75, "spo2": 99, "rr": 14, "temp": 37.8, "avpu": "A"},
            "history": "25yo female, coryzal symptoms, no difficulty swallowing, eating and drinking well."
        },
        {
            "chief_complaint": "I twisted my ankle playing football yesterday.",
            "vitals": {"hr": 72, "bp_sys": 118, "bp_dia": 72, "spo2": 99, "rr": 14, "temp": 36.8, "avpu": "A"},
            "history": "22yo male, swollen lateral ankle, can weight bear with pain, no deformity."
        },
        {
            "chief_complaint": "I've had diarrhoea and vomiting since last night.",
            "vitals": {"hr": 85, "bp_sys": 115, "bp_dia": 70, "spo2": 98, "rr": 16, "temp": 37.5, "avpu": "A"},
            "history": "35yo female, kept down fluids this morning, passing urine, no blood in stool."
        },
    ],
    
    # Category 5: Non-urgent (Blue)
    5: [
        {
            "chief_complaint": "I need a repeat prescription for my blood pressure tablets.",
            "vitals": {"hr": 70, "bp_sys": 135, "bp_dia": 85, "spo2": 99, "rr": 14, "temp": 36.7, "avpu": "A"},
            "history": "60yo male, ran out of Amlodipine, asymptomatic."
        },
        {
            "chief_complaint": "I've had a rash on my arm for a week. It's itchy.",
            "vitals": {"hr": 68, "bp_sys": 120, "bp_dia": 78, "spo2": 99, "rr": 14, "temp": 36.8, "avpu": "A"},
            "history": "30yo female, localised erythematous rash, no systemic symptoms, not spreading."
        },
        {
            "chief_complaint": "I just want my sick note signing.",
            "vitals": {"hr": 72, "bp_sys": 122, "bp_dia": 80, "spo2": 99, "rr": 14, "temp": 36.8, "avpu": "A"},
            "history": "45yo male, recovering from back strain, no red flags."
        },
    ],
}


class PatientGenerator:
    """Generates patient scenarios for the Triage environment."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self._patient_count = 0
    
    def generate(self, category: Optional[int] = None) -> Patient:
        """
        Generate a random patient.
        
        Args:
            category: Optional specific category (1-5). If None, weighted random selection.
        
        Returns:
            A Patient object.
        """
        if category is None:
            # Weighted distribution mimicking real A&E (more Cat 3-4 than Cat 1)
            weights = [5, 15, 35, 35, 10]  # % distribution
            category = random.choices([1, 2, 3, 4, 5], weights=weights)[0]
        
        scenario = random.choice(SCENARIOS[category])
        self._patient_count += 1
        
        # Add some noise to vitals
        noisy_vitals = {
            k: v + random.gauss(0, v * 0.05) if isinstance(v, float) else v
            for k, v in scenario["vitals"].items()
        }
        
        return Patient(
            id=f"P{self._patient_count:04d}",
            chief_complaint=scenario["chief_complaint"],
            vitals=noisy_vitals,
            history=scenario["history"],
            true_category=category,
            time_arrived=0,  # Will be set by environment
        )
    
    def generate_batch(self, n: int) -> List[Patient]:
        """Generate a batch of n patients."""
        return [self.generate() for _ in range(n)]


if __name__ == "__main__":
    # Quick test
    gen = PatientGenerator(seed=42)
    for _ in range(5):
        patient = gen.generate()
        print(f"{patient.id}: Cat {patient.true_category} - {patient.chief_complaint[:50]}...")
