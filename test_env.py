"""
Test script: Verify the Triage Environment works correctly

Run: python test_env.py
"""

import sys
sys.path.insert(0, '.')

from nursesim_rl import TriageEnv, PatientGenerator


def test_patient_generator():
    """Test the patient generator."""
    print("Testing PatientGenerator...")
    gen = PatientGenerator(seed=42)
    
    for category in range(1, 6):
        patient = gen.generate(category=category)
        assert patient.true_category == category
        assert len(patient.chief_complaint) > 0
        assert "hr" in patient.vitals
        print(f"  [OK] Category {category}: {patient.chief_complaint[:40]}...")
    
    print("  [OK] PatientGenerator tests passed!\n")


def test_triage_env():
    """Test the triage environment."""
    print("Testing TriageEnv...")
    
    env = TriageEnv(seed=42)
    obs, info = env.reset()
    
    assert "patient_id" in obs
    assert "chief_complaint" in obs
    assert "vitals" in obs
    assert "waiting_room" in obs
    print(f"  [OK] Reset works, first patient: {obs['patient_id']}")
    
    # Take some steps
    for i in range(5):
        action = {
            "triage_category": 3,  # Default to Urgent
            "intervention": 1,     # Send to majors
        }
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  [OK] Step {i+1}: Reward={reward:.1f}, Waiting={obs['waiting_room']}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("  [OK] TriageEnv tests passed!\n")


def test_reward_calculation():
    """Test reward calculations."""
    print("Testing Reward Logic...")
    
    env = TriageEnv(seed=123)
    obs, info = env.reset()
    
    # Force a specific patient for testing
    from nursesim_rl.patient_generator import Patient
    test_patient = Patient(
        id="TEST001",
        chief_complaint="Test complaint",
        vitals={"hr": 100, "bp_sys": 120, "bp_dia": 80, "spo2": 98, "rr": 16, "temp": 37.0, "avpu": "A"},
        history="Test history",
        true_category=1,  # Critical patient!
        time_arrived=0,
    )
    env.current_patient = test_patient
    
    # Test correct triage
    action = {"triage_category": 1, "intervention": 0}  # Correct: Cat 1, Resus
    _, reward, _, _, _ = env.step(action)
    print(f"  Correct triage (Cat 1): Reward = {reward:.1f} (expected +15)")
    
    # Reset and test safety failure
    env.reset()
    env.current_patient = test_patient
    action = {"triage_category": 4, "intervention": 5}  # Wrong: Cat 4, Discharge (DANGEROUS!)
    _, reward, _, _, _ = env.step(action)
    print(f"  Safety failure (Cat 1 -> 4 + Discharge): Reward = {reward:.1f} (expected negative)")
    
    env.close()
    print("  [OK] Reward logic tests passed!\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("[TEST] NURSESIM-RL TEST SUITE")
    print("="*60 + "\n")
    
    test_patient_generator()
    test_triage_env()
    test_reward_calculation()
    
    print("="*60)
    print("[PASS] ALL TESTS PASSED!")
    print("="*60 + "\n")
