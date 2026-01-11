---
title: NurseSim Triage
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# NurseSim-RL: A Healthcare Agent Environment for Clinical Triage

[![AgentBeats A2A](https://img.shields.io/badge/AgentBeats-A2A%20Enabled-purple)](https://agentbeats.dev/ClinyQAi/nursesim-triage)

[![OpenEnv Challenge](https://img.shields.io/badge/OpenEnv-Challenge%202026-blue)](https://rdi.berkeley.edu/agentx-agentbeats)
[![Hugging Face Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/NurseCitizenDeveloper/NurseSim-Triage-Llama-3.2-3B)
[![W&B Report](https://img.shields.io/badge/W%26B-Report-orange)](https://wandb.ai/mrlincs-nursing-citizen-development/huggingface)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **OpenEnv Challenge Entry** | Berkeley RDI AgentX-AgentBeats Competition  
> A Gymnasium-compatible RL environment for training AI agents to perform clinical triage using the Manchester Triage System (MTS).

![NurseSim Demo](docs/demo.gif)

## 🎯 Overview

**NurseSim-RL** simulates the decision-making process of a Triage Nurse in an Accident & Emergency (A&E) department. The agent must assess patients based on their chief complaint and vital signs, then assign an appropriate triage category (1-5) according to the Manchester Triage System.

### Key Features
- **Gymnasium-Compatible:** Standard RL interface for easy integration.
- **Realistic Scenarios:** 15+ patient archetypes across all 5 MTS categories.
- **Safety-Aware Rewards:** Heavy penalties for under-triaging critical patients.
- **Fine-Tuned Agent:** Llama 3.2 3B trained with Unsloth (4-bit QLoRA).

## 🏗️ Project Structure

```
NurseSim-RL/
├── nursesim_rl/           # Core environment package
│   ├── __init__.py
│   ├── TriageEnv.py       # Gymnasium environment
│   └── PatientGenerator.py # Synthetic patient generation
├── notebooks/
│   └── NurseSim_RL_Unsloth_Training.ipynb  # Training notebook
├── data/
│   ├── train.jsonl        # Training dataset (500 examples)
│   └── val.jsonl          # Validation dataset (100 examples)
├── app.py                 # Gradio demo application
├── Dockerfile             # For reproducibility
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/NurseCitizenDeveloper/NurseSim-RL.git
cd NurseSim-RL
pip install -r requirements.txt
```

### Using the Environment

```python
import gymnasium as gym
from nursesim_rl import TriageEnv

env = gym.make("NurseSim-Triage-v0")
obs, info = env.reset()

# Agent takes an action
action = {"triage_category": 2, "intervention": 1}
obs, reward, terminated, truncated, info = env.step(action)
```

### Running the Demo

**Gradio Mode (Human UI):**
```bash
export AGENT_MODE=gradio
export HF_TOKEN=your_hf_token_here
python app.py
```

**AgentBeats A2A Mode (Platform Integration):**
```bash
export AGENT_MODE=a2a
export HF_TOKEN=your_hf_token_here
python agent_main.py
```

## 🤖 AgentBeats Integration

This agent is fully compatible with the [AgentBeats platform](https://agentbeats.org) for automated agent evaluation via the **Agent-to-Agent (A2A) protocol**.

### Dual-Mode Architecture

The agent supports two deployment modes:

| Mode | Purpose | Entry Point | Port |
|------|---------|-------------|------|
| **Gradio** | Human-facing UI for demos | `app.py` | 7860 |
| **A2A** | Platform integration for automated evaluation | `agent_main.py` | 8080 |

Set the mode via the `AGENT_MODE` environment variable.

### A2A Protocol Compliance

- **Agent Card:** `.well-known/agent-card.json` - Metadata and schemas
- **Task Processing:** Structured input/output for triage assessments
- **Lifecycle Methods:** `reset()`, `health_check()`
- **Protocol Version:** A2A v1.0

### Local Testing with AgentBeats Controller

```bash
# Install earthshaker SDK
pip install earthshaker

# Set environment variables
export HF_TOKEN=your_hf_token_here
export AGENT_MODE=a2a

# Run the controller
earthshaker run_ctrl

# Test the agent card endpoint (in another terminal)
curl http://localhost:8080/.well-known/agent-card.json | jq

# Submit a test task via A2A protocol
curl -X POST http://localhost:8080/task \
  -H "Content-Type: application/json" \
  -d '{
    "complaint": "Chest pain and shortness of breath",
    "vitals": {
      "heart_rate": 120,
      "blood_pressure": "85/55",
      "spo2": 89,
      "temperature": 37.8
    }
  }'
```

### Docker Deployment

**Build:**
```bash
docker build -t nursesim-triage:latest .
```

**Run in A2A Mode:**
```bash
docker run -e HF_TOKEN=$HF_TOKEN -e AGENT_MODE=a2a -p 8080:8080 nursesim-triage:latest
```

**Run in Gradio Mode:**
```bash
docker run -e HF_TOKEN=$HF_TOKEN -e AGENT_MODE=gradio -p 7860:7860 nursesim-triage:latest
```

## 📊 Training Results

The agent was fine-tuned using **Unsloth** on a Llama 3.2 3B base model:

| Metric | Value |
|--------|-------|
| Final Loss | ~0.08 |
| Training Steps | 100 |
| Epochs | 6+ |
| Hardware | NVIDIA A100 (Colab) |

See our [W&B Report](https://wandb.ai/mrlincs-nursing-citizen-development/huggingface) for detailed training curves.

## 🩺 Clinical Framework: Manchester Triage System

| Category | Priority | Target Time | Example |
|----------|----------|-------------|---------|
| 1 | Immediate | 0 min | Cardiac arrest, Anaphylaxis |
| 2 | Very Urgent | 10 min | Chest pain, Stroke |
| 3 | Urgent | 60 min | Abdominal pain, Fractures |
| 4 | Standard | 120 min | Minor injuries, Mild illness |
| 5 | Non-Urgent | 240 min | Minor cuts, GP-suitable |

## 🔗 Links

- **Hugging Face Model:** [NurseCitizenDeveloper/NurseSim-Triage-Llama-3.2-3B](https://huggingface.co/NurseCitizenDeveloper/NurseSim-Triage-Llama-3.2-3B)
- **Gradio Demo:** [HF Spaces](https://huggingface.co/spaces/NurseCitizenDeveloper/NurseSim-Triage-Demo)
- **Training Notebook:** [Colab](notebooks/NurseSim_RL_Unsloth_Training.ipynb)

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

- **OpenEnv Challenge** - Berkeley RDI, PyTorch, Hugging Face, Unsloth
- **Manchester Triage System** - Clinical framework
- **Unsloth AI** - 2x faster fine-tuning

---

**Built for the OpenEnv Challenge 2026** 🏆
