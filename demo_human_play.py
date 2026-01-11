"""
Demo script: Play the Triage Environment as a Human

Run this to test the environment interactively.
"""

import sys
sys.path.insert(0, '.')

from nursesim_rl import TriageEnv


def main():
    env = TriageEnv(render_mode="human", seed=42)
    obs, info = env.reset()
    
    print("\n🏥 Welcome to the A&E Triage Simulator!")
    print("You are the Triage Nurse. Assess each patient and assign a category.\n")
    
    total_reward = 0
    step = 0
    
    while True:
        # Render current patient
        env.render()
        
        if obs["patient_id"] == "":
            print("\n✅ Shift complete! No more patients.")
            break
        
        # Get user input
        try:
            category = int(input("\nEnter triage category (1-5): "))
            if category < 1 or category > 5:
                print("Invalid category. Please enter 1-5.")
                continue
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        
        print("\nInterventions:")
        for i, intervention in enumerate(env.INTERVENTIONS):
            print(f"  [{i}] {intervention}")
        
        try:
            intervention_idx = int(input("Choose intervention (0-6): "))
            if intervention_idx < 0 or intervention_idx >= len(env.INTERVENTIONS):
                intervention_idx = 0
        except ValueError:
            intervention_idx = 0
        
        # Take action
        action = {
            "triage_category": category,
            "intervention": intervention_idx,
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        # Feedback
        true_cat = info.get("true_category")
        if true_cat and category == true_cat:
            print(f"\n✅ Correct! Category {category} was right. Reward: +{reward:.1f}")
        elif true_cat:
            print(f"\n⚠️  The correct category was {true_cat}. You chose {category}. Reward: {reward:.1f}")
        
        if terminated or truncated:
            break
    
    # Final stats
    print("\n" + "="*60)
    print("📊 SHIFT SUMMARY")
    print("="*60)
    print(f"  Patients Seen: {info.get('patients_seen', step)}")
    print(f"  Correct Triage: {info.get('correct_triage', 0)}")
    print(f"  Safety Failures: {info.get('safety_failures', 0)}")
    print(f"  Total Reward: {total_reward:.1f}")
    print("="*60)
    
    env.close()


if __name__ == "__main__":
    main()
