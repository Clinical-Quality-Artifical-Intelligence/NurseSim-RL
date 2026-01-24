"""
Training Dataset Generator for NurseSim-RL

Generates a dataset of triage scenarios with expert decisions for SFT training.
Output format: JSONL compatible with Unsloth/TRL.
"""

import json
import random
from typing import Dict, List
from pathlib import Path

# Import from our environment
import sys
sys.path.insert(0, str(Path(__file__).parent))
from nursesim_rl.patient_generator import PatientGenerator, SCENARIOS


def format_observation(patient_data: Dict) -> str:
    """Format patient data as a text observation for the LLM."""
    vitals = patient_data["vitals"]
    return f"""PATIENT PRESENTING TO A&E TRIAGE

Chief Complaint: "{patient_data['complaint']}"

Vitals:
- HR: {vitals['hr']:.0f} bpm
- BP: {vitals['bp_sys']:.0f}/{vitals['bp_dia']:.0f} mmHg
- SpO2: {vitals['spo2']:.0f}%
- RR: {vitals['rr']:.0f} /min
- Temp: {vitals['temp']:.1f}C
- AVPU: {vitals['avpu']}

History: {patient_data['history']}

WAITING ROOM: 12 patients | AVAILABLE BEDS: 4

What is your triage decision?"""


def get_expert_decision(category: int) -> Dict:
    """Get the expert triage decision based on category."""
    decisions = {
        1: {
            "category": 1,
            "category_name": "Immediate (Red)",
            "intervention": "send_to_resus",
            "reasoning": "Life-threatening presentation requiring immediate resuscitation. Activate trauma/medical emergency team."
        },
        2: {
            "category": 2,
            "category_name": "Very Urgent (Orange)",
            "intervention": "send_to_majors",
            "reasoning": "Time-critical condition. Requires senior review within 10 minutes. Prioritise assessment."
        },
        3: {
            "category": 3,
            "category_name": "Urgent (Yellow)",
            "intervention": "send_to_majors",
            "reasoning": "Urgent presentation requiring assessment within 60 minutes. Monitor for deterioration."
        },
        4: {
            "category": 4,
            "category_name": "Standard (Green)",
            "intervention": "send_to_minors",
            "reasoning": "Stable presentation suitable for minor injuries/illness stream. Can wait safely."
        },
        5: {
            "category": 5,
            "category_name": "Non-urgent (Blue)",
            "intervention": "refer_to_gp",
            "reasoning": "Non-urgent presentation. Redirect to primary care or self-care advice."
        },
    }
    return decisions[category]


def format_response(decision: Dict) -> str:
    """Format the expert decision as an LLM response."""
    return f"""TRIAGE DECISION:

Category: {decision['category']} - {decision['category_name']}
Intervention: {decision['intervention']}

Clinical Reasoning: {decision['reasoning']}"""


def generate_dataset(n_samples: int = 500, seed: int = 42) -> List[Dict]:
    """Generate a training dataset of triage scenarios."""
    random.seed(seed)
    dataset = []
    
    # Distribution adjusted for balanced IMMEDIATE case training
    # Original: {1: 0.05, 2: 0.15, 3: 0.35, 4: 0.35, 5: 0.10}
    category_weights = {1: 0.20, 2: 0.20, 3: 0.25, 4: 0.25, 5: 0.10}
    
    for i in range(n_samples):
        # Weighted category selection
        category = random.choices(
            list(category_weights.keys()),
            weights=list(category_weights.values())
        )[0]
        
        # Get a random scenario for this category
        scenario = random.choice(SCENARIOS[category])
        
        # Add some noise to vitals
        noisy_vitals = {}
        for k, v in scenario["vitals"].items():
            if isinstance(v, (int, float)) and k != "avpu":
                noise = random.gauss(0, abs(v) * 0.05) if v != 0 else 0
                noisy_vitals[k] = v + noise
            else:
                noisy_vitals[k] = v
        
        patient_data = {
            "complaint": scenario["chief_complaint"],
            "vitals": noisy_vitals,
            "history": scenario["history"],
        }
        
        # Format as instruction-following example
        observation = format_observation(patient_data)
        decision = get_expert_decision(category)
        response = format_response(decision)
        
        # Alpaca/ChatML format
        example = {
            "instruction": "You are an expert A&E Triage Nurse using the Manchester Triage System. Assess the following patient and provide your triage decision with clinical reasoning.",
            "input": observation,
            "output": response,
            "category": category,  # For analysis
        }
        
        dataset.append(example)
    
    return dataset


def save_dataset(dataset: List[Dict], output_path: str):
    """Save dataset to JSONL format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"[OK] Saved {len(dataset)} examples to {output_path}")


def main():
    print("\n" + "="*60)
    print("[DATASET] NurseSim-RL Training Data Generator")
    print("="*60 + "\n")
    
    # Generate training set
    print("Generating training dataset (500 examples)...")
    train_data = generate_dataset(n_samples=500, seed=42)
    save_dataset(train_data, "data/train.jsonl")
    
    # Generate validation set
    print("Generating validation dataset (100 examples)...")
    val_data = generate_dataset(n_samples=100, seed=123)
    save_dataset(val_data, "data/val.jsonl")
    
    # Stats
    print("\n" + "-"*40)
    print("Dataset Statistics:")
    for cat in range(1, 6):
        train_count = sum(1 for x in train_data if x["category"] == cat)
        val_count = sum(1 for x in val_data if x["category"] == cat)
        print(f"  Category {cat}: {train_count} train / {val_count} val")
    print("-"*40 + "\n")
    
    # Preview
    print("Sample training example:")
    print("-"*40)
    sample = train_data[0]
    print(f"[INSTRUCTION]\n{sample['instruction']}\n")
    print(f"[INPUT]\n{sample['input']}\n")
    print(f"[OUTPUT]\n{sample['output']}")
    print("-"*40 + "\n")


if __name__ == "__main__":
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    main()
