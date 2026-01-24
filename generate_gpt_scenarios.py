"""
GPT-Powered Triage Scenario Generator
Generates gold-standard synthetic training data using GPT-5/4
"""

import os
import json
import openai
from pathlib import Path

# Try to get API key from environment
API_KEY = os.environ.get('OPENAI_API_KEY')
if not API_KEY:
    print("⚠️ Please set OPENAI_API_KEY environment variable")
    print("Run: $env:OPENAI_API_KEY='your-key-here'")
    exit(1)

client = openai.OpenAI(api_key=API_KEY)

# Find working model
MODELS = ['gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo']
MODEL = None

print("🔍 Finding available model...")
for m in MODELS:
    try:
        client.chat.completions.create(model=m, messages=[{"role": "user", "content": "test"}], max_tokens=5)
        MODEL = m
        print(f"✅ Using: {m}")
        break
    except Exception as e:
        continue

if not MODEL:
    print("❌ No model available")
    exit(1)

# System prompt for clinical accuracy
SYSTEM_PROMPT = """You are an expert A&E Triage Nurse and clinical educator with 20 years of experience.
You are creating realistic training scenarios for the Manchester Triage System.

For each scenario, provide a JSON object with:
1. chief_complaint: What the patient/relative actually SAYS (patient language, not medical jargon)
2. vitals: Object with hr, bp_sys, bp_dia, spo2, rr, temp, avpu (A/V/P/U)
3. history: Brief clinical history (age, gender, relevant PMH, time course)

CRITICAL RULES:
- Vitals MUST be physiologically consistent (shock = low BP + high HR)
- Chief complaints use patient language, not medical terminology
- Include diverse demographics
- Cover atypical presentations

Return a JSON array of 10 scenarios. No markdown, just JSON."""

CATEGORY_PROMPTS = {
    1: """Generate 10 Category 1 (IMMEDIATE/Red) triage scenarios.
MTS Discriminators: Airway compromise, Inadequate breathing, Shock, Unresponsive, Currently fitting

Include: STEMI, stroke, anaphylaxis, trauma, septic shock, cardiac arrest, status epilepticus, 
massive GI bleed, tension pneumothorax, DKA coma, eclampsia, meningococcal sepsis, drowning, burns with inhalation.""",
    
    2: """Generate 10 Category 2 (VERY URGENT/Orange) scenarios.
MTS Discriminators: Severe pain, Altered consciousness, Very hot adult/child, Significant mechanism

Include: Chest pain (possible ACS), thunderclap headache, focal neurology, high fever with red flags, 
significant trauma, acute abdomen, spinal injury, ongoing seizure that stopped.""",
    
    3: """Generate 10 Category 3 (URGENT/Yellow) scenarios.
MTS Discriminators: Moderate pain, Hot adult/child, Persistent vomiting, Pleuritic pain

Include: COPD exacerbation, cellulitis, renal colic, fractures, moderate asthma, 
acute confusion (elderly), DVT symptoms, pyelonephritis.""",
    
    4: """Generate 10 Category 4 (STANDARD/Green) scenarios.
MTS Discriminators: Recent mild pain, Warm, Recent problem

Include: Minor injuries, viral illnesses, stable chronic conditions, minor lacerations, 
sprains, insect bites, UTI symptoms, ear infections.""",
    
    5: """Generate 10 Category 5 (NON-URGENT/Blue) scenarios.
MTS Discriminators: Recent mild problem

Include: Prescription requests, chronic stable issues, minor rashes, medication reviews, 
sick notes, minor aches lasting weeks."""
}


def generate_batch(category: int, batch_num: int) -> list:
    """Generate a batch of scenarios for a category"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": CATEGORY_PROMPTS[category] + f"\n\nBatch {batch_num} - ensure unique scenarios not seen before."}
            ],
            temperature=0.9,
            max_tokens=4000
        )
        
        content = response.choices[0].message.content.strip()
        # Clean markdown
        if content.startswith('```'):
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"    ⚠️ JSON Error. Attempting repair...")
            # Simple repair: try to find the list bracket
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1:
                try:
                    return json.loads(content[start:end+1])
                except:
                    pass
            print(f"    ❌ Failed to parse JSON batch.")
            return []
            
    except Exception as e:
        print(f"    Error: {str(e)[:60]}")
        return []


def format_training_example(scenario: dict, category: int) -> dict:
    """Convert scenario to training format"""
    vitals = scenario.get('vitals', {})
    
    input_text = f"""PATIENT PRESENTING TO A&E TRIAGE

Chief Complaint: "{scenario.get('chief_complaint', 'Unknown')}"

Vitals:
- HR: {vitals.get('hr', 80):.0f} bpm
- BP: {vitals.get('bp_sys', 120):.0f}/{vitals.get('bp_dia', 80):.0f} mmHg
- SpO2: {vitals.get('spo2', 98):.0f}%
- RR: {vitals.get('rr', 16):.0f} /min
- Temp: {vitals.get('temp', 37.0):.1f}C
- AVPU: {vitals.get('avpu', 'A')}

History: {scenario.get('history', 'Unknown')}

WAITING ROOM: 12 patients | AVAILABLE BEDS: 4

What is your triage decision?"""

    decisions = {
        1: ("Immediate (Red)", "send_to_resus", "Life-threatening presentation requiring immediate resuscitation."),
        2: ("Very Urgent (Orange)", "send_to_majors", "Time-critical condition. Requires senior review within 10 minutes."),
        3: ("Urgent (Yellow)", "send_to_majors", "Urgent presentation requiring assessment within 60 minutes."),
        4: ("Standard (Green)", "send_to_minors", "Stable presentation suitable for minor injuries/illness stream."),
        5: ("Non-urgent (Blue)", "refer_to_gp", "Non-urgent presentation. Redirect to primary care.")
    }
    
    name, action, reason = decisions[category]
    output_text = f"""TRIAGE DECISION:

Category: {category} - {name}
Intervention: {action}

Clinical Reasoning: {reason}"""

    return {
        "instruction": "You are an expert A&E Triage Nurse using the Manchester Triage System. Assess the following patient and provide your triage decision with clinical reasoning.",
        "input": input_text,
        "output": output_text,
        "category": category
    }


def main():
    print("\n" + "="*60)
    print("🏥 GPT-Powered Triage Scenario Generator")
    print("="*60 + "\n")
    
    # Configure batches per category
    batches = {1: 5, 2: 3, 3: 3, 4: 3, 5: 2}
    
    all_scenarios = {}
    training_data = []
    
    for cat in [1, 2, 3, 4, 5]:
        print(f"\n📋 Category {cat} ({batches[cat]} batches):")
        all_scenarios[cat] = []
        
        for b in range(batches[cat]):
            print(f"  Batch {b+1}...", end=" ")
            scenarios = generate_batch(cat, b+1)
            all_scenarios[cat].extend(scenarios)
            print(f"✅ {len(scenarios)} scenarios")
        
        # Convert to training format
        for s in all_scenarios[cat]:
            try:
                example = format_training_example(s, cat)
                training_data.append(example)
            except:
                continue
        
        print(f"  Total Cat {cat}: {len(all_scenarios[cat])} scenarios")
    
    # Save raw scenarios
    with open('data/gpt_scenarios.json', 'w') as f:
        json.dump(all_scenarios, f, indent=2)
    print(f"\n✅ Saved raw scenarios: data/gpt_scenarios.json")
    
    # Save as JSONL with unique timestamp
    import time
    timestamp = int(time.time())
    output_file = f'data/gpt_train_{timestamp}.jsonl'
    
    with open(output_file, 'w') as f:
        for ex in training_data:
            f.write(json.dumps(ex) + '\n')
    print(f"✅ Saved training data: {output_file} ({len(training_data)} examples)")
    
    # skipped automatic merging to avoid race conditions
    print(f"✅ Saved batch: {output_file} ({len(training_data)} examples)")
    
    print("\n" + "="*60)
    print("✅ Generation complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
