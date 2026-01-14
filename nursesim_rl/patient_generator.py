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
# Gold-standard scenarios validated against MTS discriminators and clinical guidelines
SCENARIOS = {
    # Category 1: Immediate (Red) - Life threatening
    # MTS Discriminators: Airway compromise, Inadequate breathing, Shock, Unresponsive
    1: [
        # Original scenarios
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
        # NEW: STEMI - Classic presentation
        {
            "chief_complaint": "There's an elephant sitting on my chest... I feel like I'm going to die.",
            "vitals": {"hr": 110, "bp_sys": 80, "bp_dia": 55, "spo2": 91, "rr": 28, "temp": 36.8, "avpu": "A"},
            "history": "58yo male, sweating profusely, nausea, pain radiating to jaw and left arm, 30 mins onset."
        },
        # NEW: Acute Stroke - FAST positive
        {
            "chief_complaint": "My husband's face has dropped and he can't speak properly!",
            "vitals": {"hr": 95, "bp_sys": 195, "bp_dia": 110, "spo2": 96, "rr": 18, "temp": 37.0, "avpu": "V"},
            "history": "70yo male, sudden onset 45 mins ago, right-sided weakness, slurred speech, facial droop."
        },
        # NEW: Tension Pneumothorax
        {
            "chief_complaint": "I was stabbed... I can't breathe... everything is going dark.",
            "vitals": {"hr": 135, "bp_sys": 75, "bp_dia": 45, "spo2": 78, "rr": 40, "temp": 36.5, "avpu": "V"},
            "history": "25yo male, stab wound to right chest, trachea deviated left, absent breath sounds right."
        },
        # NEW: Septic Shock
        {
            "chief_complaint": "She's been confused all day and now she's going cold and clammy.",
            "vitals": {"hr": 125, "bp_sys": 70, "bp_dia": 40, "spo2": 88, "rr": 30, "temp": 39.8, "avpu": "V"},
            "history": "78yo female, recent UTI, mottled skin, delayed capillary refill, drowsy."
        },
        # NEW: Major Trauma - MVA
        {
            "chief_complaint": "He was thrown from the car... he's not making sense!",
            "vitals": {"hr": 130, "bp_sys": 85, "bp_dia": 50, "spo2": 90, "rr": 32, "temp": 35.5, "avpu": "V"},
            "history": "35yo male, unrestrained driver, high-speed RTC, GCS 9, deformed left femur."
        },
        # NEW: Massive GI Bleed
        {
            "chief_complaint": "I've vomited loads of blood and now I feel faint...",
            "vitals": {"hr": 140, "bp_sys": 75, "bp_dia": 45, "spo2": 94, "rr": 26, "temp": 36.2, "avpu": "V"},
            "history": "55yo male, known liver disease, haematemesis x3, melena, pale and diaphoretic."
        },
        # NEW: Status Epilepticus
        {
            "chief_complaint": "My daughter won't stop fitting! It's been going on for ages!",
            "vitals": {"hr": 145, "bp_sys": 160, "bp_dia": 95, "spo2": 82, "rr": 8, "temp": 38.5, "avpu": "U"},
            "history": "8yo female, known epilepsy, continuous tonic-clonic seizure for 12 minutes, cyanosed."
        },
        # NEW: DKA with Coma
        {
            "chief_complaint": "He's a diabetic and now he won't wake up properly.",
            "vitals": {"hr": 115, "bp_sys": 90, "bp_dia": 55, "spo2": 95, "rr": 35, "temp": 37.2, "avpu": "P"},
            "history": "22yo male, Type 1 diabetic, Kussmaul breathing, fruity breath, missed insulin for 3 days."
        },
        # NEW: Eclampsia
        {
            "chief_complaint": "She's 8 months pregnant and started fitting!",
            "vitals": {"hr": 120, "bp_sys": 180, "bp_dia": 120, "spo2": 89, "rr": 24, "temp": 37.5, "avpu": "U"},
            "history": "32yo female, 34 weeks pregnant, seizure witnessed, severe headache and visual disturbance earlier."
        },
        # NEW: Complete Airway Obstruction
        {
            "chief_complaint": "He was eating steak and now he can't breathe at all!",
            "vitals": {"hr": 140, "bp_sys": 160, "bp_dia": 100, "spo2": 65, "rr": 0, "temp": 37.0, "avpu": "A"},
            "history": "60yo male, choking at restaurant, cannot cough or speak, universal choking sign, cyanotic."
        },
        # NEW: Haemorrhagic Stroke (SAH)
        {
            "chief_complaint": "It's the worst headache of my life... like being hit with a hammer.",
            "vitals": {"hr": 55, "bp_sys": 200, "bp_dia": 115, "spo2": 94, "rr": 14, "temp": 37.3, "avpu": "V"},
            "history": "48yo female, sudden onset thunderclap headache, vomiting, photophobia, now drowsy."
        },
        # NEW: Ruptured AAA
        {
            "chief_complaint": "The pain in my back is unbearable... I feel like I'm going to pass out.",
            "vitals": {"hr": 125, "bp_sys": 75, "bp_dia": 50, "spo2": 93, "rr": 28, "temp": 36.0, "avpu": "V"},
            "history": "72yo male, known AAA, sudden severe abdominal/back pain, pulsatile mass, pale and sweating."
        },
        # NEW: Atypical ACS (Elderly)
        {
            "chief_complaint": "I just feel really unwell... something is very wrong.",
            "vitals": {"hr": 45, "bp_sys": 80, "bp_dia": 50, "spo2": 88, "rr": 24, "temp": 36.5, "avpu": "V"},
            "history": "82yo female, known diabetes, vague malaise, nausea, cool and clammy, no chest pain."
        },
        # NEW: Meningococcal Septicaemia
        {
            "chief_complaint": "My child has a rash that won't go away when I press it!",
            "vitals": {"hr": 180, "bp_sys": 70, "bp_dia": 40, "spo2": 90, "rr": 45, "temp": 40.2, "avpu": "V"},
            "history": "4yo male, non-blanching purpuric rash, headache, photophobia, neck stiffness, drowsy."
        },
        # NEW: Drowning/Near-drowning
        {
            "chief_complaint": "He was pulled from the pool and he's not breathing properly!",
            "vitals": {"hr": 50, "bp_sys": 70, "bp_dia": 45, "spo2": 60, "rr": 6, "temp": 34.0, "avpu": "U"},
            "history": "6yo male, submersion ~5 mins, bystander CPR given, now gasping, hypothermic."
        },
        # NEW: Severe Burns with Inhalation
        {
            "chief_complaint": "She was in a house fire... her voice sounds strange now.",
            "vitals": {"hr": 130, "bp_sys": 90, "bp_dia": 55, "spo2": 85, "rr": 32, "temp": 37.5, "avpu": "A"},
            "history": "45yo female, facial burns, singed nasal hairs, hoarse voice, stridor developing."
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
