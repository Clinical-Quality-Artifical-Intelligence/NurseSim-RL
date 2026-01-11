#!/usr/bin/env python3
"""
NurseSim-Triage A2A Agent Entry Point

This module implements the Agent-to-Agent (A2A) protocol interface for NurseSim-Triage.
It creates a persistent FastAPI server to handle requests from the AgentBeats platform.
"""

import os
import json
import torch
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ==========================================
# Data Models
# ==========================================

class Vitals(BaseModel):
    heart_rate: int = 80
    blood_pressure: str = "120/80"
    spo2: int = 98
    temperature: float = 37.0

class TaskInput(BaseModel):
    complaint: str
    vitals: Vitals

# ==========================================
# Agent Core Logic
# ==========================================

class NurseSimTriageAgent:
    """
    A2A-compatible triage agent for clinical assessment.
    """
    
    def __init__(self):
        """Initialize the triage agent placeholder."""
        self.model = None
        self.tokenizer = None
        self.HF_TOKEN = os.environ.get("HF_TOKEN")
        
        if not self.HF_TOKEN:
            print("WARNING: HF_TOKEN not set. Model loading will fail if authentication is required.")
    
    async def load_model(self):
        """Load the base model and LoRA adapters asynchronously."""
        if self.model is not None:
            return

        try:
            print("⏳ Starting model load...")
            base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
            adapter_id = "NurseCitizenDeveloper/NurseSim-Triage-Llama-3.2-3B"
            
            # Offload heavy loading to thread to avoid blocking event loop
            await asyncio.to_thread(self._load_weights, base_model_id, adapter_id)
            
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ CRITICAL ERROR loading model: {e}")
            import traceback
            traceback.print_exc()

    def _load_weights(self, base_model_id, adapter_id):
        print(f"Loading tokenizer from {adapter_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_id,
            token=self.HF_TOKEN
        )
        
        print(f"Loading base model {base_model_id} with 4-bit quantization...")
        
        # Modern 4-bit loading configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=self.HF_TOKEN,
        )
        
        print(f"Applying LoRA adapters from {adapter_id}...")
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_id,
            token=self.HF_TOKEN
        )
        self.model.eval()

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process an A2A task and return the triage assessment."""
        if self.model is None:
            return {
                "error": "ModelStillLoading", 
                "message": "The agent is still warming up. Please retry in 30 seconds."
            }

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
            
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            
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
        if "immediate" in response_lower or "resuscitation" in response_lower: return "Immediate"
        elif "very urgent" in response_lower or "emergency" in response_lower: return "Very Urgent"
        elif "urgent" in response_lower: return "Urgent"
        elif "standard" in response_lower: return "Standard"
        elif "non-urgent" in response_lower or "non urgent" in response_lower: return "Non-Urgent"
        else: return "Standard"
    
    def _extract_recommended_action(self, response: str) -> str:
        if "monitor" in response.lower(): return "Monitor patient closely"
        elif "immediate" in response.lower() or "urgent" in response.lower(): return "Immediate medical attention required"
        else: return "Continue assessment and treatment as per protocol"
    
    def health_check(self) -> Dict[str, Any]:
        """Return agent health status."""
        return {
            "status": "healthy" if self.model is not None else "loading",
            "model_loaded": self.model is not None,
            "gpu_available": torch.cuda.is_available()
        }

# ==========================================
# FastAPI Lifecycle & App
# ==========================================

agent = NurseSimTriageAgent()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model in background
    print("🚀 Server starting. Triggering model load task...")
    asyncio.create_task(agent.load_model())
    yield
    # Shutdown logic (if any)
    print("🛑 Server shutting down.")

app = FastAPI(
    title="NurseSim-Triage Agent",
    version="1.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "NurseSim-Triage Agent Online",
        "status": agent.health_check()["status"]
    }

@app.get("/health")
async def health_check():
    return agent.health_check()

@app.get("/.well-known/agent-card.json")
async def get_agent_card():
    card_path = ".well-known/agent-card.json"
    if os.path.exists(card_path):
        with open(card_path, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Agent card not found")

@app.post("/process-task")
async def process_task(task: TaskInput):
    result = agent.process_task(task.dict())
    if "error" in result and result.get("message") == "ModelStillLoading":
        raise HTTPException(status_code=503, detail=result["message"])
    return result

# ==========================================
# Entry Point
# ==========================================

if __name__ == "__main__":
    print("Starting A2A Server on port 7860...")
    uvicorn.run(app, host="0.0.0.0", port=7860)
