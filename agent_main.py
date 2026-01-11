#!/usr/bin/env python3
"""
NurseSim-Triage A2A Agent Entry Point

This module implements the Agent-to-Agent (A2A) protocol interface for NurseSim-Triage.
It enables integration with the AgentBeats platform for automated agent evaluation.
"""

import os
import json
import torch
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class NurseSimTriageAgent:
    """
    A2A-compatible triage agent for clinical assessment.
    
    This agent uses a fine-tuned Llama 3.2 3B model to perform emergency
    department triage based on patient complaints and vital signs.
    """
    
    def __init__(self):
        """Initialize the triage agent and load the model."""
        self.model = None
        self.tokenizer = None
        self.HF_TOKEN = os.environ.get("HF_TOKEN")
        
        if not self.HF_TOKEN:
            raise ValueError("HF_TOKEN environment variable must be set")
        
        self._load_model()
    
    def _load_model(self):
        """Load the base model and LoRA adapters."""
        if self.model is not None:
            return  # Already loaded
        
        base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
        adapter_id = "NurseCitizenDeveloper/NurseSim-Triage-Llama-3.2-3B"
        
        print(f"Loading tokenizer from {adapter_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_id,
            token=self.HF_TOKEN
        )
        
        print(f"Loading base model {base_model_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
            token=self.HF_TOKEN,
        )
        
        print(f"Applying LoRA adapters from {adapter_id}...")
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_id,
            token=self.HF_TOKEN
        )
        self.model.eval()
        print("Model loaded successfully!")
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an A2A task and return the triage assessment.
        
        Args:
            task: Dictionary containing:
                - complaint (str): Patient's chief complaint
                - vitals (dict): Dictionary with heart_rate, blood_pressure, spo2, temperature
        
        Returns:
            Dictionary containing:
                - triage_category (str): MTS category
                - assessment (str): Clinical assessment
                - recommended_action (str): Next steps
        """
        try:
            # Extract task data
            complaint = task.get("complaint", "")
            vitals = task.get("vitals", {})
            
            hr = vitals.get("heart_rate", 80)
            bp = vitals.get("blood_pressure", "120/80")
            spo2 = vitals.get("spo2", 98)
            temp = vitals.get("temperature", 37.0)
            
            # Create prompt
            prompt = f"""### Instruction:
You are an expert A&E Triage Nurse. Assess the following patient and provide your triage decision.

### Input:
Patient Complaint: {complaint}
Vitals: HR {hr}, BP {bp}, SpO2 {spo2}%, Temp {temp}C.

### Response:"""
            
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response after the ### Response: marker
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            
            # Parse the response to extract triage category
            triage_category = self._extract_triage_category(response)
            
            return {
                "triage_category": triage_category,
                "assessment": response,
                "recommended_action": self._extract_recommended_action(response)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "triage_category": "Error",
                "assessment": f"Error processing task: {str(e)}"
            }
    
    def _extract_triage_category(self, response: str) -> str:
        """Extract the triage category from the model's response."""
        response_lower = response.lower()
        
        # Check for MTS categories in order of urgency
        if "immediate" in response_lower or "resuscitation" in response_lower:
            return "Immediate"
        elif "very urgent" in response_lower or "emergency" in response_lower:
            return "Very Urgent"
        elif "urgent" in response_lower:
            return "Urgent"
        elif "standard" in response_lower:
            return "Standard"
        elif "non-urgent" in response_lower or "non urgent" in response_lower:
            return "Non-Urgent"
        else:
            return "Standard"  # Default
    
    def _extract_recommended_action(self, response: str) -> str:
        """Extract recommended actions from the model's response."""
        # Look for common action keywords
        if "monitor" in response.lower():
            return "Monitor patient closely"
        elif "immediate" in response.lower() or "urgent" in response.lower():
            return "Immediate medical attention required"
        else:
            return "Continue assessment and treatment as per protocol"
    
    def reset(self):
        """Reset the agent state (A2A lifecycle method)."""
        # For stateless agents, this is a no-op
        # If we add conversation history, clear it here
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """Return agent health status."""
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "gpu_available": torch.cuda.is_available(),
            "device": str(self.model.device) if self.model else "not loaded"
        }


def main():
    """
    Main entry point for the A2A agent.
    
    This function demonstrates how the agent would be used in an A2A context.
    In production, this would be managed by the AgentBeats controller.
    """
    print("Initializing NurseSim-Triage A2A Agent...")
    agent = NurseSimTriageAgent()
    
    # Example task
    example_task = {
        "complaint": "Crushing chest pain and nausea",
        "vitals": {
            "heart_rate": 110,
            "blood_pressure": "90/60",
            "spo2": 94,
            "temperature": 37.2
        }
    }
    
    print("\n" + "="*60)
    print("Processing example task...")
    print(f"Input: {json.dumps(example_task, indent=2)}")
    print("="*60 + "\n")
    
    result = agent.process_task(example_task)
    
    print("Result:")
    print(json.dumps(result, indent=2))
    print("\n" + "="*60)
    
    # Health check
    health = agent.health_check()
    print("\nAgent Health:")
    print(json.dumps(health, indent=2))


if __name__ == "__main__":
    main()
