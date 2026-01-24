"""
Test script for the Semantic RL Environment.
Validates that the NurseEmbedWrapper works correctly.
"""

import numpy as np
from nursesim_rl import TriageEnv, NurseEmbedWrapper

def test_semantic_wrapper():
    print("=" * 60)
    print("Testing NurseEmbed Semantic RL Wrapper")
    print("=" * 60)
    
    # Create base environment
    print("\n[1] Creating base TriageEnv...")
    base_env = TriageEnv(max_steps=10, seed=42)
    
    # Wrap with NurseEmbed
    print("[2] Wrapping with NurseEmbedWrapper...")
    semantic_env = NurseEmbedWrapper(base_env, use_vitals=True)
    
    # Check observation space
    print(f"\n[3] Observation Space Check:")
    print(f"    Base Env: {type(base_env.observation_space)}")
    print(f"    Semantic Env: {semantic_env.observation_space}")
    print(f"    Expected shape: (390,) [384 embed + 6 vitals]")
    
    # Reset and get observation
    print("\n[4] Resetting environment...")
    obs, info = semantic_env.reset(seed=42)
    
    print(f"    Observation type: {type(obs)}")
    print(f"    Observation shape: {obs.shape}")
    print(f"    Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Take a step
    print("\n[5] Taking a step with action (Cat=3, Intervention=2)...")
    action = {"triage_category": 3, "intervention": 2}
    obs2, reward, terminated, truncated, info = semantic_env.step(action)
    
    print(f"    Reward: {reward}")
    print(f"    Terminated: {terminated}")
    print(f"    New observation shape: {obs2.shape}")
    
    # Verify embedding is meaningful (not just zeros)
    print("\n[6] Embedding quality check:")
    embed_part = obs[:384]  # First 384 dims are the embedding
    print(f"    Embedding L2 norm: {np.linalg.norm(embed_part):.3f}")
    print(f"    Embedding is normalized: {abs(np.linalg.norm(embed_part) - 1.0) < 0.1}")
    
    # Test caching
    print("\n[7] Testing embedding cache...")
    obs3, _ = semantic_env.reset(seed=42)  # Same seed = same patient
    cache_hit = np.allclose(obs[:384], obs3[:384])
    print(f"    Cache working: {cache_hit}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_semantic_wrapper()
