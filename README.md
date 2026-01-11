# NurseSim-RL: A Healthcare Agent Environment for Clinical Triage

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

```bash
python app.py
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
