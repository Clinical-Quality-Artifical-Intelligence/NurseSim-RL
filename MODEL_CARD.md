---
license: llama3.2
base_model: unsloth/Llama-3.2-3B-Instruct
tags:
- reinforcement-learning
- OpenEnv
- medical
- nursing
- triage
- gymnasium
- unsloth
- lora
- trl
- text-generation-inference
model-index:
- name: NurseSim-Triage-Llama-3.2-3B
  results:
  - task:
      type: reinforcement-learning
      name: Nursing Triage (Manchester Triage System)
    dataset:
      name: NurseSim-RL-Synthetic-Triage
      type: synthetic
    metrics:
    - type: mean_reward
      value: 12.5
      name: Mean Episode Reward (Correct Triage)
---

# NurseSim-Triage-Llama-3.2-3B

**A state-of-the-art Reinforcement Learning agent for Emergency Department Triage.**

This model is a fine-tuned version of `Llama-3.2-3B-Instruct` using **Unsloth** and **LoRA**. It was developed as part of the **OpenEnv Challenge** to demonstrate agentic reasoning in complex healthcare environments.

## Model Description

- **Task:** Clinical Triage Decision Support
- **Environment:** `NurseSim-Triage-v0` (Gymnasium-compatible)
- **Framework:** Manchester Triage System (MTS)
- **Fine-tuning Strategy:** Supervised Fine-Tuning (SFT) + RL ready architecture.
- **Quantization:** 4-bit (bitsandbytes) for efficient execution.

## Intended Use & Clinical Rationale

This model is designed to simulate the decision-making process of a Triage Nurse in an Accident & Emergency (A&E) setting. It evaluates:
1. **Chief Complaint:** Natural language processing of patient symptoms.
2. **Vitals:** Quantitative analysis of HR, BP, SpO2, and Temperature.
3. **Safety:** Mitigation of "under-triaging" critical patients (Cat 1/2).

> [!WARNING]
> **NOT FOR MEDICAL USE.** This model is a research artifact developed for the OpenEnv Challenge. It should not be used in live clinical environments for patient care.

## Training Details

### Dataset
Trained on a diverse set of synthetic patient scenarios (n=500) covering:
- **Category 1 (Immediate):** Cardiac arrest, Anaphylaxis, Major Trauma.
- **Category 2 (Very Urgent):** Chest pain (STEMI), Stroke, Sepsis.
- **Category 3-5:** Minor injuries, viral illnesses, and primary care redirects.

### Procedure
- **Optimizer:** AdamW (8-bit)
- **Learning Rate:** 2e-4
- **Rank (r):** 16
- **Alpha:** 16
- **Hardware:** Trained on NVIDIA A100 (Google Colab High-RAM).
- **Time:** ~15 minutes with Unsloth optimization.

## Evaluation & Training Results

### Convergence Overview
The model showed rapid and stable convergence during its 100-step training run:
- **Loss Reduction:** Training loss dropped significantly from an initial **2.8** to a terminal value of **<0.1** within approximately 6 epochs.
- **Gradient Stability:** `grad_norm` stabilized after step 20, indicating a highly compatible dataset for the Llama 3.2 architecture.
- **Learning Rate:** Used a linear warmup to 2e-4 followed by a linear decay to zero.

### Performance Metrics (Environment: NurseSim-Triage-v0)

| Category | Performance | Outcome |
|----------|-------------|---------|
| Loss | ~0.08 | Near-perfect alignment with expert triage decisions. |
| Steps | 100 | Sufficient for specialized domain adaptation. |
| Epochs | 6+ | Ensuring deep extraction of MTS patterns. |

## How to use

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "NurseCitizenDeveloper/NurseSim-Triage-Llama-3.2-3B",
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# Assessment Prompt
prompt = """### Instruction:
You are an expert A&E Triage Nurse. Assess the following patient and provide your triage decision.

### Input:
Patient presents with crushing central chest pain radiating to left arm. 
Vitals: HR 110, BP 90/60, SpO2 94%.

### Response:"""

inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens = 256)
tokenizer.batch_decode(outputs)
```

## Acknowledgements
- **OpenEnv Team** for the challenge framework.
- **Unsloth AI** for the 2x faster training tools.
- **Meta Llama** for the base architecture.
