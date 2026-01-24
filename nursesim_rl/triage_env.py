"""
TriageEnv: A Gymnasium-compatible RL environment for A&E Triage.

OpenEnv Challenge Entry - 2026
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, Optional, Tuple

from .patient_generator import PatientGenerator, Patient


class TriageEnv(gym.Env):
    """
    A&E Triage Environment.
    
    The agent plays the role of a Triage Nurse, assessing patients and
    assigning them to the correct Manchester Triage System category.
    
    Observation:
        - patient_complaint (str): The patient's chief complaint
        - vitals (dict): HR, BP, SpO2, RR, Temp, AVPU
        - history (str): Brief clinical history
        - waiting_room (int): Number of patients currently waiting
        - available_beds (int): Beds available in Resus/Majors
    
    Action:
        - triage_category (int): 1-5 (Immediate to Non-urgent)
        - intervention (str): One of the allowed interventions
    
    Reward:
        - +10 for correct triage category
        - +5 for adjacent category (within 1)
        - -50 for critical safety failure (under-triaging P1/P2 by 2+ levels)
        - -1 per minute waiting for high-acuity patients
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}
    
    INTERVENTIONS = [
        "send_to_resus",
        "send_to_majors", 
        "send_to_minors",
        "order_ecg",
        "give_analgesia",
        "discharge",
        "refer_to_gp",
    ]
    
    def __init__(
        self,
        max_patients: int = 20,
        max_steps: int = 50,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.max_patients = max_patients
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        self.patient_generator = PatientGenerator(seed=seed)
        
        # Action space: Discrete triage category + intervention
        self.action_space = spaces.Dict({
            "triage_category": spaces.Discrete(5, start=1),  # 1-5
            "intervention": spaces.Discrete(len(self.INTERVENTIONS)),
        })
        
        # Observation space
        self.observation_space = spaces.Dict({
            "patient_id": spaces.Text(10),
            "chief_complaint": spaces.Text(500),
            "vitals": spaces.Dict({
                "hr": spaces.Box(0, 300, shape=(), dtype=np.float32),
                "bp_sys": spaces.Box(0, 300, shape=(), dtype=np.float32),
                "bp_dia": spaces.Box(0, 200, shape=(), dtype=np.float32),
                "spo2": spaces.Box(0, 100, shape=(), dtype=np.float32),
                "rr": spaces.Box(0, 60, shape=(), dtype=np.float32),
                "temp": spaces.Box(30, 45, shape=(), dtype=np.float32),
                "avpu": spaces.Text(1),
            }),
            "history": spaces.Text(500),
            "waiting_room": spaces.Discrete(100),
            "available_beds": spaces.Discrete(20),
        })
        
        # State
        self.current_patient: Optional[Patient] = None
        self.waiting_queue: list = []
        self.step_count: int = 0
        self.total_reward: float = 0.0
        self.available_beds: int = 10
        self.episode_stats: Dict[str, Any] = {}
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            self.patient_generator = PatientGenerator(seed=seed)
        
        # Reset state
        self.step_count = 0
        self.total_reward = 0.0
        self.available_beds = 10
        self.episode_stats = {
            "correct_triage": 0,
            "safety_failures": 0,
            "patients_seen": 0,
        }
        
        # Generate initial waiting room
        initial_patients = np.random.randint(3, 8)
        self.waiting_queue = self.patient_generator.generate_batch(initial_patients)
        for i, p in enumerate(self.waiting_queue):
            p.time_arrived = -i * 5  # Stagger arrival times
        
        # Get first patient
        self.current_patient = self._get_next_patient()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Dict with 'triage_category' (1-5) and 'intervention' (index)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        
        if self.current_patient is None:
            # No more patients - episode ends
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Parse action
        assigned_category = action.get("triage_category", 3)
        intervention_idx = action.get("intervention", 0)
        intervention = self.INTERVENTIONS[intervention_idx]
        
        # Calculate reward
        reward = self._calculate_reward(assigned_category, intervention)
        self.total_reward += reward
        self.episode_stats["patients_seen"] += 1
        
        # Update bed availability based on intervention
        if intervention in ["send_to_resus", "send_to_majors"]:
            self.available_beds = max(0, self.available_beds - 1)
        elif intervention in ["discharge", "refer_to_gp"]:
            self.available_beds = min(10, self.available_beds + 1)
        
        # Possibly add new patients to queue
        if np.random.random() < 0.3:  # 30% chance of new arrival
            new_patient = self.patient_generator.generate()
            new_patient.time_arrived = self.step_count
            self.waiting_queue.append(new_patient)
        
        # Get next patient
        self.current_patient = self._get_next_patient()
        
        # Check termination
        terminated = self.current_patient is None and len(self.waiting_queue) == 0
        truncated = self.step_count >= self.max_steps
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _calculate_reward(self, assigned_category: int, intervention: str) -> float:
        """Calculate reward based on triage decision."""
        if self.current_patient is None:
            return 0.0
        
        true_category = self.current_patient.true_category
        category_diff = abs(assigned_category - true_category)
        
        reward = 0.0
        
        # Category accuracy
        if category_diff == 0:
            reward += 10.0
            self.episode_stats["correct_triage"] += 1
        elif category_diff == 1:
            reward += 5.0  # Close enough
        else:
            reward -= 5.0 * category_diff  # Penalty scales with error
        
        # Critical safety failure: Under-triaging a critical patient
        if true_category <= 2 and assigned_category >= true_category + 2:
            reward -= 50.0
            self.episode_stats["safety_failures"] += 1
        
        # Intervention appropriateness
        if true_category == 1 and intervention == "send_to_resus":
            reward += 5.0
        elif true_category == 5 and intervention in ["discharge", "refer_to_gp"]:
            reward += 3.0
        elif true_category == 1 and intervention == "discharge":
            reward -= 30.0  # Never discharge a P1!
        
        return reward
    
    def _get_next_patient(self) -> Optional[Patient]:
        """Get the next patient from the queue (FIFO with priority override)."""
        if not self.waiting_queue:
            return None
        
        # Priority override: P1 patients jump the queue
        for i, patient in enumerate(self.waiting_queue):
            if patient.true_category == 1:
                return self.waiting_queue.pop(i)
        
        # Otherwise FIFO
        return self.waiting_queue.pop(0)
    
    def _get_observation(self) -> Dict:
        """Build the observation dictionary."""
        if self.current_patient is None:
            return {
                "patient_id": "",
                "chief_complaint": "No patients waiting.",
                "vitals": {
                    "hr": 0.0, "bp_sys": 0.0, "bp_dia": 0.0,
                    "spo2": 0.0, "rr": 0.0, "temp": 0.0, "avpu": "A"
                },
                "history": "",
                "waiting_room": len(self.waiting_queue),
                "available_beds": self.available_beds,
            }
        
        return {
            "patient_id": self.current_patient.id,
            "chief_complaint": self.current_patient.chief_complaint,
            "vitals": {
                "hr": float(self.current_patient.vitals.get("hr", 0)),
                "bp_sys": float(self.current_patient.vitals.get("bp_sys", 0)),
                "bp_dia": float(self.current_patient.vitals.get("bp_dia", 0)),
                "spo2": float(self.current_patient.vitals.get("spo2", 0)),
                "rr": float(self.current_patient.vitals.get("rr", 0)),
                "temp": float(self.current_patient.vitals.get("temp", 0)),
                "avpu": str(self.current_patient.vitals.get("avpu", "A")),
            },
            "history": self.current_patient.history,
            "waiting_room": len(self.waiting_queue),
            "available_beds": self.available_beds,
        }
    
    def _get_info(self) -> Dict:
        """Return additional info."""
        return {
            "step": self.step_count,
            "total_reward": self.total_reward,
            "true_category": self.current_patient.true_category if self.current_patient else None,
            **self.episode_stats,
        }
    
    def render(self) -> Optional[str]:
        """Render the environment."""
        if self.render_mode == "human" or self.render_mode == "ansi":
            obs = self._get_observation()
            output = f"""
╔══════════════════════════════════════════════════════════════════╗
║  A&E TRIAGE SIMULATOR  │  Step: {self.step_count:3d} │ Waiting: {obs['waiting_room']:2d} │ Beds: {obs['available_beds']:2d}  ║
╠══════════════════════════════════════════════════════════════════╣
║  PATIENT: {obs['patient_id']:<54} ║
╠──────────────────────────────────────────────────────────────────╣
║  Chief Complaint:                                                ║
║    "{obs['chief_complaint'][:60]:<60}" ║
╠──────────────────────────────────────────────────────────────────╣
║  VITALS:                                                         ║
║    HR: {obs['vitals']['hr']:>3.0f}  │  BP: {obs['vitals']['bp_sys']:>3.0f}/{obs['vitals']['bp_dia']:<3.0f}  │  SpO2: {obs['vitals']['spo2']:>3.0f}%            ║
║    RR: {obs['vitals']['rr']:>3.0f}  │  Temp: {obs['vitals']['temp']:.1f}°C  │  AVPU: {obs['vitals']['avpu']}               ║
╠──────────────────────────────────────────────────────────────────╣
║  History: {obs['history'][:55]:<55} ║
╠══════════════════════════════════════════════════════════════════╣
║  What is your triage decision?                                   ║
║    [1] Immediate  [2] Very Urgent  [3] Urgent  [4] Std  [5] Non  ║
╚══════════════════════════════════════════════════════════════════╝
"""
            if self.render_mode == "human":
                print(output)
            return output
        return None
    
    def close(self):
        """Clean up resources."""
        pass


# Register with Gymnasium
gym.register(
    id="NurseSim-Triage-v0",
    entry_point="nursesim_rl:TriageEnv",
)
