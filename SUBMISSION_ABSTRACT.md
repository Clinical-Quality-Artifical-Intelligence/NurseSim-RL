# Submission Abstract: NurseSim-RL

## Project Name
NurseSim-RL: A Healthcare Agent Environment for Clinical Triage

## Abstract (for submission form)

NurseSim-RL is a Gymnasium-compatible reinforcement learning environment that simulates clinical triage in an Emergency Department (A&E) setting. The environment challenges AI agents to assess patients based on natural language chief complaints and vital sign data, then assign appropriate triage categories (1-5) according to the Manchester Triage System (MTS).

**Key Contributions:**
1. **Novel Healthcare RL Environment:** A safety-critical environment where incorrect decisions carry severe penalties, modeling real-world clinical risk.
2. **Synthetic Clinical Dataset:** 500+ diverse patient scenarios covering all 5 MTS categories, with realistic vital sign variations.
3. **Fine-Tuned LLM Agent:** A Llama 3.2 3B model trained using Unsloth (4-bit QLoRA) demonstrating rapid domain adaptation (2.8 → 0.08 loss in 100 steps).
4. **Reproducible Pipeline:** Complete training notebook, Dockerfile, and Gradio demo for immediate deployment.

**Evaluation Focus:** Healthcare Agent Track - The benchmark evaluates clinical reasoning, safety awareness, and resource allocation under time pressure.

**Impact:** This environment enables development and testing of AI agents for healthcare decision support, with direct applications in triage training, clinical education, and NHS workforce optimization.

---

## Suggested Answers for Form Fields

**Participation Category:** Create a new benchmark

**Evaluation Track(s):** Healthcare Agent

**Specific Benchmarks:** N/A (new benchmark)

**Demo Video Title:** "NurseSim-RL: AI Triage Agent Demo - OpenEnv Challenge 2026"
