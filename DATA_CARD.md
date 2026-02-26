# Data Card: NurseSim-Triage Dataset

> **Framework:** Google Data Cards Playbook v1.0  
> **Last Updated:** 2026-02-20  
> **Status:** Pre-Publication  

---

## Dataset Overview

| Field | Value |
|---|---|
| **Dataset Name** | NurseSim-Triage |
| **Version** | v1.0 |
| **Format** | JSONL (JSON Lines) |
| **Files** | `data/train.jsonl`, `data/test.jsonl`, `data/val.jsonl` |
| **Size (train)** | ~500 records |
| **License** | CC-BY-4.0 |
| **Language** | English |
| **Domain** | Clinical / Emergency Nursing |
| **Task** | Instruction-following, Triage Classification |

---

## Motivations & Intentions

### Why was this dataset created?

NurseSim-Triage was created to train and evaluate Large Language Models (LLMs) on **Manchester Triage System (MTS)** clinical decision-making. There is a significant gap in publicly available, high-quality datasets for emergency triage in the UK NHS context. This dataset directly addresses that gap, enabling:

- Fine-tuning of open-weight LLMs for A&E triage support
- Benchmarking AI clinical decision-making against the MTS standard
- Research into AI safety in time-critical nursing environments

### Primary Use Cases

- **Research:** LLM fine-tuning and evaluation for clinical AI research
- **Education:** Training nursing students and registered nurses on MTS triage reasoning
- **Benchmarking:** Evaluating AI model performance on structured clinical classification

### Out-of-Scope Uses

> [!WARNING]
> This dataset is **not** intended for direct clinical deployment or to make autonomous patient care decisions. It is a research and education dataset only.

- Autonomous triage without qualified nurse oversight
- Replacing registered nursing clinical judgment
- Use in jurisdictions with different triage frameworks (e.g., CTAS, ESI) without validation

---

## Dataset Composition

### Structure

Each record in the JSONL file is a JSON object with the following fields:

| Field | Type | Description |
|---|---|---|
| `instruction` | string | System prompt defining the MTS expert nurse persona |
| `input` | string | Patient scenario: chief complaint, vital signs (HR, BP, SpO2, RR, Temp, AVPU), history, and department context (waiting patients, available beds) |
| `output` | string | Triage decision: category, intervention, and clinical reasoning |
| `category` | integer | Triage category label (1–5) |

```json
{
  "instruction": "You are an expert A&E Triage Nurse using the Manchester Triage System...",
  "input": "PATIENT PRESENTING TO A&E TRIAGE\n\nChief Complaint: \"I've had abdominal pain...\"\nVitals:\n- HR: 105 bpm\n...",
  "output": "TRIAGE DECISION:\n\nCategory: 3 - Urgent (Yellow)\nIntervention: send_to_majors\n\nClinical Reasoning: ...",
  "category": 3
}
```

### Label Distribution

| Category | MTS Level | Colour | Target Count |
|---|---|---|---|
| 1 | Immediate | Red | ~100 |
| 2 | Very Urgent | Orange | ~100 |
| 3 | Urgent | Yellow | ~100 |
| 4 | Standard | Green | ~100 |
| 5 | Non-Urgent | Blue | ~100 |

### Splits

| Split | File | Records |
|---|---|---|
| Train | `data/train.jsonl` | ~500 |
| Validation | `data/val.jsonl` | TBC |
| Test | `data/test.jsonl` | TBC |

---

## Data Collection & Creation

### Collection Method

**Synthetic Generation.** All data was synthetically generated. No real patient data was used. The generation process followed the `nursesim-triage` skill, using:

1. **25 base clinical scenarios** covering canonical MTS presentations across all 5 triage categories
2. **Vital sign augmentation:** Each base scenario was varied systematically (HR ±10%, BP variations, SpO2 drops, temperature extremes) to generate diverse presentations
3. **Contextual variation:** Department busyness (waiting room count, available beds) was varied to simulate real-world triage pressure

### Clinical Validity

Scenarios were designed by a registered nurse with A&E experience and reviewed against:
- Manchester Triage System Group (2014) *Triage Nurse*
- NICE Clinical Guidelines relevant to presenting complaints
- NMC Standards of Proficiency for Registered Nurses

### Generation Tools

- **Primary LLM:** Llama-3 / Open-weight models via NVIDIA NIM or equivalent
- **Judge Model:** LLM-as-a-Judge pattern for quality filtering
- **Framework:** NVIDIA Nemotron / custom synthetic data pipeline

---

## Data Quality

### Known Limitations

- **Synthetic origin:** Scenarios may not fully capture the complexity and ambiguity of real A&E presentations
- **English only:** No multilingual support; UK English clinical terminology used throughout
- **UK NHS context:** MTS categories and interventions are aligned to NHS England A&E workflow; may not generalise to other healthcare systems
- **No comorbidity complexity:** Base scenarios are relatively clean presentations; complex comorbidities (e.g., dementia + sepsis) are underrepresented in v1.0
- **Demographic neutrality:** Synthetic patients do not carry demographic identifiers; equity across age, gender, and ethnicity has not been validated

### Quality Assurance

- All outputs include structured `TRIAGE DECISION` with category, intervention, and clinical reasoning
- Category labels are consistent with MTS definitions
- LLM-as-a-Judge filtering applied during generation to remove low-quality outputs

---

## Ethical Considerations

### Data Privacy

- **No personally identifiable information (PII):** All data is fully synthetic
- No real patients, no real NHS records, no real clinical encounters

### Potential Harms

- AI models trained on this dataset may produce incorrect triage decisions; human nurse oversight is mandatory
- Under-triage (assigning too low a category) is a patient safety risk; model outputs should always be reviewed
- The dataset may encode biases of the MTS framework itself (e.g., pain assessment bias in darker skin tones)

### Bias Considerations

- The MTS framework has known limitations in assessing pain and skin signs in patients with darker skin tones — a validated limitation this project is working to address in future versions via the EWAAST dataset
- Synthetic generation may reflect biases in the base LLM's training data

### Governance Alignment

This dataset was designed in alignment with the [GOV.UK AI Engineering Lab](https://github.com/govuk-digital-backbone/aiengineeringlab) guardrails:

| Guardrail | Alignment |
|---|---|
| **G-DH-02** — Prohibited data types | No PII, no real patient data, fully synthetic |
| **G-ET-01** — Clinical safety review trigger | Healthcare/clinical AI datasets require clinical review before deployment |
| **G-ET-03** — Human judgment requirements | Dataset outputs are advisory; RN triage decision is mandatory |
| **G-OV-01** — Factual verification | Triage categories verified against MTS definitions |

---

## Attribution & Licensing

| Field | Value |
|---|---|
| **Author** | NurseCitizenDeveloper |
| **Affiliation** | Independent NHS Nursing AI Research |
| **Hugging Face Org** | NurseCitizenDeveloper |
| **License** | Creative Commons Attribution 4.0 International (CC-BY-4.0) |
| **Citation** | TBC — manuscript in preparation |

### Citation (BibTeX — Placeholder)

```bibtex
@dataset{nursesim_triage_2026,
  author    = {NurseCitizenDeveloper},
  title     = {NurseSim-Triage: A Synthetic Dataset for Manchester Triage System AI},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/NurseCitizenDeveloper/NurseSim-Triage}
}
```

---

## Related Resources

- [NurseSim-RL Repository](https://github.com/NurseCitizenDeveloper/NurseSim-RL)
- [NurseReason-Dataset](../NurseReason-Dataset/) — clinical reasoning companion dataset
- [AI Educator Toolkit](../AI-Educator-Toolkit/) — educational documentation
- Manchester Triage Group. (2014). *Emergency Triage: Manchester Triage Group* (3rd ed.). Wiley-Blackwell.
