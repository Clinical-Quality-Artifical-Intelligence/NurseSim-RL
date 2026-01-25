#!/usr/bin/env python3
"""
NurseSim-Triage Hybrid Agent Entry Point

This module combines the A2A API (for AgentBeats) and the Gradio UI (for Human/Demo)
into a single FastAPI application listening on port 7860.
"""

import os
import json
import torch
import uvicorn
import asyncio
import gradio as gr
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from pydantic import BaseModel
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# PDS Client for NHS patient lookup
from nursesim_rl.pds_client import PDSClient, PDSEnvironment, PatientDemographics

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
    nhs_number: Optional[str] = None  # If provided, fetch demographics from PDS
    age: Optional[int] = None
    gender: Optional[str] = None

# ==========================================
# Agent Core Logic
# ==========================================

class NurseSimTriageAgent:
    """
    Shared agent logic for both API and UI.
    """
    
    def __init__(self):
        """Initialize the triage agent placeholder."""
        self.model = None
        self.tokenizer = None
        self.HF_TOKEN = os.environ.get("HF_TOKEN")
        
        # Initialize PDS client for NHS patient lookup (sandbox mode)
        self.pds_client = PDSClient(environment=PDSEnvironment.SANDBOX)
        
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
            
            # Offload heavy loading to thread
            await asyncio.to_thread(self._load_weights, base_model_id, adapter_id)
            
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ CRITICAL ERROR loading model: {e}")
            import traceback
            traceback.print_exc()

    def _load_weights(self, base_model_id, adapter_id):
        print(f"Loading tokenizer from {adapter_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_id, token=self.HF_TOKEN)
        
        print(f"Loading base model {base_model_id} with 4-bit quantization...")
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
        self.model = PeftModel.from_pretrained(self.model, adapter_id, token=self.HF_TOKEN)
        self.model.eval()

    def get_response(self, complaint: str, hr: int, bp: str, spo2: int, temp: float) -> str:
        """Shared inference logic."""
        if self.model is None:
            return "⚠️ System is warming up. Please try again in 30 seconds."

        prompt = f"""### Instruction:
You are an expert A&E Triage Nurse. Assess the following patient and provide your triage decision.

### Input:
Patient Complaint: {complaint}
Vitals: HR {hr}, BP {bp}, SpO2 {spo2}%, Temp {temp}C.

### Response:"""
        
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
            
        return response

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process an API task, optionally fetching patient demographics from PDS."""
        if self.model is None:
            return {
                "error": "ModelStillLoading", 
                "message": "The agent is still warming up. Please retry in 30 seconds."
            }

        try:
            complaint = task.get("complaint", "")
            vitals = task.get("vitals", {})
            nhs_number = task.get("nhs_number")
            
            # If NHS number provided, enrich with PDS data
            patient_info = None
            if nhs_number:
                try:
                    patient_info = self.lookup_patient(nhs_number)
                except Exception as e:
                    print(f"PDS lookup failed: {e}")
            
            response = self.get_response(
                complaint, 
                vitals.get("heart_rate", 80),
                vitals.get("blood_pressure", "120/80"),
                vitals.get("spo2", 98),
                vitals.get("temperature", 37.0)
            )
            
            result = {
                "triage_category": self._extract_triage_category(response),
                "assessment": response,
                "recommended_action": self._extract_recommended_action(response)
            }
            
            # Include patient info if retrieved
            if patient_info:
                result["patient"] = {
                    "nhs_number": patient_info.nhs_number,
                    "name": patient_info.full_name,
                    "age": patient_info.age,
                    "gender": patient_info.gender,
                    "gp_practice": patient_info.gp_practice_name,
                }
            
            return result
            
        except Exception as e:
            return {"error": str(e), "triage_category": "Error"}
    
    def lookup_patient(self, nhs_number: str) -> PatientDemographics:
        """
        Look up patient demographics from NHS PDS.
        
        Args:
            nhs_number: 10-digit NHS number
            
        Returns:
            PatientDemographics object with patient details
        """
        return self.pds_client.lookup_patient_sync(nhs_number)
    
    def _extract_triage_category(self, response: str) -> str:
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
        return {
            "status": "healthy" if self.model is not None else "loading",
            "model_loaded": self.model is not None,
            "gpu_available": torch.cuda.is_available()
        }

# ==========================================
# Application Setup
# ==========================================

agent = NurseSimTriageAgent()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server starting. Triggering model load task...")
    asyncio.create_task(agent.load_model())
    yield
    print("🛑 Server shutting down.")

app = FastAPI(title="NurseSim-Triage Agent", version="1.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# API Endpoints
# ==========================================

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

class PatientLookupRequest(BaseModel):
    nhs_number: str

@app.post("/lookup-patient")
async def api_lookup_patient(request: PatientLookupRequest):
    """Direct endpoint to lookup patient details from NHS PDS."""
    try:
        patient = agent.lookup_patient(request.nhs_number)
        return {
            "nhs_number": patient.nhs_number,
            "full_name": patient.full_name,
            "date_of_birth": patient.date_of_birth,
            "age": patient.age,
            "gender": patient.gender,
            "address": patient.address,
            "gp_practice": patient.gp_practice_name
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Gradio UI Integration
# ==========================================

def gradio_predict(complaint, hr, bp, spo2, temp):
    return agent.get_response(complaint, hr, bp, spo2, temp)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🩺 NurseSim AI: Emergency Triage Simulator
    **An AI agent fine-tuned for the Manchester Triage System (MTS).**
    *Developed for the OpenEnv Challenge by NurseCitizenDeveloper.*
    
    > ⚡ **Hybrid Mode**: Serving both Gradio UI and A2A API (AgentBeats)
    """)
    
    with gr.Row():
        with gr.Column():
            complaint = gr.Textbox(label="Chief Complaint", placeholder="e.g., Shortness of breath...")
            with gr.Row():
                hr = gr.Number(label="Heart Rate", value=80)
                bp = gr.Textbox(label="Blood Pressure", placeholder="e.g., 120/80")
            with gr.Row():
                spo2 = gr.Slider(label="SpO2 (%)", minimum=50, maximum=100, value=98)
                temp = gr.Number(label="Temperature (C)", value=37.0)
            
            submit_btn = gr.Button("Assess Patient", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="AI Triage Assessment", lines=10)
            gr.Markdown("### ⚠️ Research Prototype - Not for Clinical Use")

    submit_btn.click(gradio_predict, inputs=[complaint, hr, bp, spo2, temp], outputs=output_text)
    
    gr.Examples(
        examples=[
            ["Crushing chest pain and nausea", 110, "90/60", 94, 37.2],
            ["Twisted ankle at football", 75, "125/85", 99, 36.8],
        ],
        inputs=[complaint, hr, bp, spo2, temp]
    )

# Mount Gradio app to FastAPI at root
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    print("Starting Hybrid Server on port 7860...")
    uvicorn.run(app, host="0.0.0.0", port=7860)
