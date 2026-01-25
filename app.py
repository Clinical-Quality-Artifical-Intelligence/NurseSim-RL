import gradio as gr
import spaces
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from nursesim_rl.pds_client import lookup_patient_sync

# Get HF token from environment (set as a Space secret)
HF_TOKEN = os.environ.get("HF_TOKEN")

# Global model/tokenizer
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
        adapter_id = "NurseCitizenDeveloper/NurseSim-Triage-Llama-3.2-3B"
        
        tokenizer = AutoTokenizer.from_pretrained(adapter_id, token=HF_TOKEN)
        
        # Load base model in 4-bit
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
            token=HF_TOKEN,
        )
        # Apply LoRA adapters
        model = PeftModel.from_pretrained(model, adapter_id, token=HF_TOKEN)
        model.eval()
    return model, tokenizer

def lookup_patient_ui(nhs_no):
    """Gradio handler for PDS lookup."""
    if not nhs_no:
        return 45, "Male", "", "Please enter an NHS Number."
        
    try:
        patient = lookup_patient_sync(nhs_no)
        # Format PMH with GP info as context
        pmh_context = f"Registered GP: {patient.gp_practice_name}"
        status_msg = f"✅ Verified: {patient.full_name}"
        return patient.age, patient.gender, pmh_context, status_msg
    except Exception as e:
        return 45, "Male", "", f"❌ Lookup failed: {str(e)}"

def format_prompt(complaint, hr, bp, spo2, temp, rr, avpu, age, gender, pmh):
    # Construct History Dictionary (Critical for Model Accuracy)
    history_dict = {
        'age': int(age) if age else "Unknown",
        'gender': gender,
        'relevant_PMH': pmh if pmh else "None",
        'time_course': "See complaint"
    }
    
    # Exact Training Data Format
    input_text = f"""PATIENT PRESENTING TO A&E TRIAGE

Chief Complaint: "{complaint}"

Vitals:
- HR: {hr} bpm
- BP: {bp} mmHg
- SpO2: {spo2}%
- RR: {rr} /min
- Temp: {temp}C
- AVPU: {avpu}

History: {history_dict}

WAITING ROOM: 12 patients | AVAILABLE BEDS: 4

What is your triage decision?"""

    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert A&E Triage Nurse using the Manchester Triage System. Assess the following patient and provide your triage decision with clinical reasoning.

### Input:
{input_text}

### Response:
"""

@spaces.GPU(duration=120)
def triage_patient(complaint, age, gender, pmh, hr, bp, spo2, rr, temp, avpu):
    model, tokenizer = load_model()
    
    prompt = format_prompt(complaint, hr, bp, spo2, temp, rr, avpu, age, gender, pmh)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown("""
    # 🏥 NurseSim AI: Emergency Triage Simulator
    **An AI agent fine-tuned for the Manchester Triage System (MTS).**
    
    > **Note:** This model is trained to be **Age-Aware**. A 72-year-old with chest pain is treated differently than a 20-year-old.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Patient Demographics")
            
            with gr.Row():
                nhs_number = gr.Textbox(label="NHS Number", placeholder="e.g. 9000000009", scale=2)
                lookup_btn = gr.Button("🔍 Lookup", variant="secondary", scale=1)
            
            lookup_status = gr.Markdown("")
            
            age = gr.Number(label="Age", value=45)
            gender = gr.Radio(["Male", "Female"], label="Gender", value="Male")
            pmh = gr.Textbox(label="Medical History (PMH)", placeholder="e.g., Hypertension, Diabetes, Asthma", lines=2)
            
            gr.Markdown("### 2. Presentation")
            complaint = gr.Textbox(label="Chief Complaint", placeholder="e.g., Crushing chest pain radiating to jaw", lines=2)
            
        with gr.Column(scale=1):
            gr.Markdown("### 3. Vital Signs")
            with gr.Row():
                hr = gr.Number(label="HR (bpm)", value=80)
                rr = gr.Number(label="RR (breaths/min)", value=16)
            with gr.Row():
                bp = gr.Textbox(label="BP (mmHg)", value="120/80")
                spo2 = gr.Slider(label="SpO2 (%)", minimum=50, maximum=100, value=98)
            with gr.Row():
                temp = gr.Number(label="Temp (C)", value=37.0)
                avpu = gr.Dropdown(["A", "V", "P", "U"], label="AVPU", value="A")
            
            submit_btn = gr.Button("🚨 Assess Patient", variant="primary", size="lg")
            
    with gr.Row():
        output_text = gr.Textbox(label="AI Triage Assessment", lines=8, show_copy_button=True)
        
    gr.Markdown("""
    ### ⚠️ Safety Disclaimer
    This system is a **research prototype** developed for the OpenEnv Challenge. 
    It is **NOT** a certified medical device and should not be used for real clinical decision-making.
    """)

    # Wire up lookup button
    lookup_btn.click(
        fn=lookup_patient_ui,
        inputs=[nhs_number],
        outputs=[age, gender, pmh, lookup_status]
    )

    submit_btn.click(
        fn=triage_patient,
        inputs=[complaint, age, gender, pmh, hr, bp, spo2, rr, temp, avpu],
        outputs=output_text
    )
    
    gr.Examples(
        examples=[
            ["Crushing chest pain and nausea", 72, "Male", "HTN, High Cholesterol", 110, "90/60", 94, 24, 37.2, "A"],
            ["Twisted ankle at football", 22, "Male", "None", 75, "125/85", 99, 14, 36.8, "A"],
            ["Swollen tongue after peanuts", 25, "Female", "Nut Allergy", 120, "90/60", 91, 28, 37.5, "A"],
        ],
        inputs=[complaint, age, gender, pmh, hr, bp, spo2, rr, temp, avpu],
        label="Test Scenarios"
    )

if __name__ == "__main__":
    demo.launch()
