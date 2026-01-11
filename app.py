import gradio as gr
import spaces
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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
            token=HF_TOKEN,  # Pass token for gated model access
        )
        # Apply LoRA adapters
        model = PeftModel.from_pretrained(model, adapter_id, token=HF_TOKEN)
        model.eval()
    return model, tokenizer

@spaces.GPU(duration=120)
def triage_patient(complaint, hr, bp, spo2, temp):
    model, tokenizer = load_model()
    
    prompt = f"""### Instruction:
You are an expert A&E Triage Nurse. Assess the following patient and provide your triage decision.

### Input:
Patient Complaint: {complaint}
Vitals: HR {hr}, BP {bp}, SpO2 {spo2}%, Temp {temp}C.

### Response:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🩺 NurseSim AI: Emergency Triage Simulator
    **An AI agent fine-tuned for the Manchester Triage System (MTS).**
    *Developed for the OpenEnv Challenge by NurseCitizenDeveloper.*
    
    > ⚡ Powered by **ZeroGPU** - Model loads on-demand.
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
            gr.Markdown("""
            ### ⚠️ Safety Warning
            This is a research prototype. **NOT** a certified medical device.
            """)

    submit_btn.click(
        fn=triage_patient,
        inputs=[complaint, hr, bp, spo2, temp],
        outputs=output_text
    )
    
    gr.Examples(
        examples=[
            ["Crushing chest pain and nausea", 110, "90/60", 94, 37.2],
            ["Twisted ankle at football", 75, "125/85", 99, 36.8],
            ["High fever and confusion", 105, "100/70", 92, 39.5],
        ],
        inputs=[complaint, hr, bp, spo2, temp]
    )

if __name__ == "__main__":
    demo.launch()
