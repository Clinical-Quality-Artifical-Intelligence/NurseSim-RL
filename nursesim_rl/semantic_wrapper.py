"""
NurseEmbedWrapper: A Gymnasium wrapper that converts text observations to NurseEmbed vectors.

This enables Language-Conditioned Reinforcement Learning for nursing scenarios.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, Tuple

# Lazy load NurseEmbed to avoid import errors if not available
_embed_model = None

def _get_embed_model():
    """Lazy load the NurseEmbed model."""
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer("NurseCitizenDeveloper/NurseEmbed-300M")
            print("[OK] NurseEmbed model loaded successfully")
        except Exception as e:
            print(f"[WARN] NurseEmbed not available: {e}")
            # Fallback to random embeddings for testing
            _embed_model = "fallback"
    return _embed_model


class NurseEmbedWrapper(gym.Wrapper):
    """
    Wraps a TriageEnv and converts text observations to NurseEmbed vectors.
    Also flattens the Dict action space to MultiDiscrete for SB3 compatibility.
    
    Instead of the agent seeing:
        {"chief_complaint": "Chest pain radiating to arm...", "vitals": {...}, ...}
    
    It sees:
        np.array([...390 dimensional semantic vector...])
    
    The vector encodes the MEANING of the clinical presentation.
    """
    
    EMBED_DIM = 384  # NurseEmbed-300M outputs 384D vectors (nomic base)
    
    def __init__(self, env: gym.Env, use_vitals: bool = True):
        """
        Args:
            env: The underlying TriageEnv
            use_vitals: Whether to append vitals to the embedding
        """
        super().__init__(env)
        
        self.use_vitals = use_vitals
        self.model = _get_embed_model()
        
        # Calculate observation dimension
        obs_dim = self.EMBED_DIM
        if use_vitals:
            obs_dim += 6  # HR, BP_sys, BP_dia, SpO2, RR, Temp
        
        # Override observation space to be a flat Box
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Flatten action space for SB3 compatibility
        # Original: Dict({'triage_category': Discrete(5, start=1), 'intervention': Discrete(7)})
        # New: MultiDiscrete([5, 7]) where first dim is category (0-4, add 1 later) and second is intervention
        self.action_space = spaces.MultiDiscrete([5, 7])
        
        # Cache for embeddings (same text -> same embedding)
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset and convert observation."""
        obs, info = self.env.reset(**kwargs)
        return self._convert_observation(obs), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Convert action, step, convert observation."""
        # Convert flat action [category_idx, intervention_idx] to Dict
        dict_action = {
            "triage_category": int(action[0]) + 1,  # 0-4 -> 1-5
            "intervention": int(action[1])
        }
        obs, reward, terminated, truncated, info = self.env.step(dict_action)
        return self._convert_observation(obs), reward, terminated, truncated, info
    
    def _convert_observation(self, obs: Dict) -> np.ndarray:
        """Convert Dict observation to semantic vector."""
        # Build text representation
        text = self._build_clinical_text(obs)
        
        # Get embedding (with caching)
        embedding = self._get_embedding(text)
        
        # Optionally append vitals
        if self.use_vitals:
            vitals_vector = self._extract_vitals(obs)
            embedding = np.concatenate([embedding, vitals_vector])
        
        return embedding.astype(np.float32)
    
    def _build_clinical_text(self, obs: Dict) -> str:
        """Build a clinical description from the observation."""
        complaint = obs.get("chief_complaint", "Unknown complaint")
        history = obs.get("history", "")
        
        # Create a rich clinical description
        text = f"Patient presents with: {complaint}. Clinical history: {history}."
        
        # Add vitals context as text for semantic understanding
        vitals = obs.get("vitals", {})
        if vitals:
            text += f" Vital signs: HR {vitals.get('hr', 'N/A')}, "
            text += f"BP {vitals.get('bp_sys', 'N/A')}/{vitals.get('bp_dia', 'N/A')}, "
            text += f"SpO2 {vitals.get('spo2', 'N/A')}%, "
            text += f"RR {vitals.get('rr', 'N/A')}, "
            text += f"AVPU {vitals.get('avpu', 'A')}."
        
        return text
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        if self.model == "fallback":
            # Fallback: deterministic pseudo-random embedding based on text hash
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.randn(self.EMBED_DIM)
        else:
            embedding = self.model.encode(text, normalize_embeddings=True)
        
        self._embedding_cache[text] = embedding
        return embedding
    
    def _extract_vitals(self, obs: Dict) -> np.ndarray:
        """Extract vitals as a normalized vector."""
        vitals = obs.get("vitals", {})
        return np.array([
            vitals.get("hr", 70) / 200.0,  # Normalize HR
            vitals.get("bp_sys", 120) / 200.0,
            vitals.get("bp_dia", 80) / 150.0,
            vitals.get("spo2", 98) / 100.0,
            vitals.get("rr", 16) / 40.0,
            (vitals.get("temp", 37.0) - 35) / 5.0,  # Normalize temp
        ], dtype=np.float32)


def make_semantic_triage_env(seed: int = None, **kwargs) -> gym.Env:
    """Factory function to create a semantically-aware triage environment."""
    from nursesim_rl import TriageEnv
    
    base_env = TriageEnv(seed=seed, **kwargs)
    wrapped_env = NurseEmbedWrapper(base_env, use_vitals=True)
    
    return wrapped_env


# Register wrapped version
gym.register(
    id="NurseSim-SemanticTriage-v0",
    entry_point="nursesim_rl.semantic_wrapper:make_semantic_triage_env",
)
