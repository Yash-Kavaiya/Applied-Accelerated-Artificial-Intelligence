# Week 12: Healthcare AI Applications (Part 1 Summary)

## The central idea

Healthcare AI is **not a chatbot or a single model call**. It is a safety-critical, auditable decision-support system. The pipeline is a loop, not a one-off prediction:

**Data sources (EHR text, labs, radiology images, wearables) → Preprocessing → AI model → Calibrated prediction → Clinician review → Audited action → feedback/monitoring back into the system**

**Safety principle:** the system should make the clinician *more informed, not less responsible*. The correct framing is decision support with human review, never autonomous replacement of clinicians.

## Why healthcare AI is different

- **Errors hurt people directly.** A false positive can cause unnecessary tests or treatment. A false negative can delay care. So you must think about the *cost of each error type*, not just average accuracy.
- **Clinical data is difficult.** It's sensitive, incomplete, delayed, biased by who receives care, and collected for care rather than for ML. Even *missingness* can carry clinical meaning.
- **Outputs must fit the workflow.** Who receives the alert, when, can they act on it, is the evidence visible, can they override, and is it logged?

A clinical AI output should answer: *What evidence supports this? How uncertain is it? Who reviews it? What is logged?*

<img width="1448" height="1086" alt="image" src="https://github.com/user-attachments/assets/509d5cfc-8498-4c74-aef1-17da8401c422" />

## Data modalities

| Modality | Examples | AI tasks |
|---|---|---|
| Structured EHR | demographics, medications, diagnoses, vitals, labs | risk prediction, cohort selection, quality measures |
| Clinical text | notes, discharge summaries, radiology reports | summarization, information extraction, Q&A |
| Medical images | X-ray, CT, MRI, ultrasound, pathology | classification, detection, segmentation, report assistance |
| Signals/devices | ECG, ICU waveforms, wearables | arrhythmia detection, early warning, monitoring |
| Administrative | claims, appointments, utilization | resource planning, readmission analysis |

Different modalities need different models and preprocessing (e.g., CNN/ViT for images, language model for notes, tabular model for structured risk).

**A patient is multimodal.** Two patients can have similar-looking X-rays, but one has stable vitals and mild symptoms while the other has hypoxia, fever and comorbidities. So healthcare AI increasingly combines text, images, tables and time series. More data helps, but it also raises privacy risk, integration complexity, bias and validation burden. Answers should be grounded in **patient facts, guidelines, local protocols and clinician judgement**.

## FHIR (Fast Healthcare Interoperability Resources)

- An HL7 standard for exchanging healthcare data electronically.
- Uses **modular resources**: Patient, Observation, Condition, MedicationRequest, Encounter, DiagnosticReport.
- Makes clinical context structured, discoverable and machine-processable. An agent should read from controlled resources rather than scrape screens or free text.
- The demo uses a simplified FHIR-like JSON record, e.g. an `Observation` with code `oxygen-saturation` and value 91%.

## Part 1: Medical imaging AI

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/d9a43258-a4da-4be0-a9ce-3d19017845d3" />


**Workflow:** DICOM/metadata → preprocessing → CNN/transformer → heatmap/mask → radiologist review → structured report (with logging and auditability).

**Pipeline stages**
- **Acquisition:** device and protocol determine resolution, field of view, artifacts. Scanners, protocols and positioning change the image distribution.
- **De-identification:** remove or transform patient identifiers before research or demo use.
- **Preprocessing:** resampling, windowing, normalization, cropping, quality checks.
- **Inference:** probability, heatmap, mask, measurement or draft text.
- **Review:** specialist checks output against clinical context, uncertainty and failure cases.

**Common tasks**

| Task | Output | Example |
|---|---|---|
| Classification | label/probability | pneumonia likely/unlikely |
| Detection | bounding box | lung nodule, fracture, hemorrhage candidate |
| Segmentation | pixel/voxel mask | organ volume, tumor boundary |
| Triage | priority flag | urgent review for suspected stroke |
| Report assistance | draft impression | summarize measurements and findings |

*"AI detected a region" ≠ "final diagnosis."*

**Failure examples:** wrong orientation, scanner protocol shift, out-of-distribution anatomy, artifacts (e.g., metal) mistaken for disease, shortcut features learned instead of disease, and plausible-looking heatmaps that aren't causal evidence. Hence the teaching point: **the model is only one stage of the imaging system**, and validation must go beyond one clean dataset.

**Explanations (heatmaps): useful but limited**
- They show which regions influenced a prediction and help clinicians review quickly.
- But a heatmap is **not proof** the model used a medically valid feature. It may focus on a confounder or marker and still look plausible.
- ✅ Useful question: *"Does the highlighted region correspond to a plausible clinical finding?"*
- ❌ Dangerous question: *"The heatmap looks medical, so the model must be correct."*
- Interpretability supports review but is **not a substitute for external validation**.

## Part 2: EHR risk prediction and clinical decision support

**Chain:** EHR/FHIR → risk score → calibration → explanation → decision support.

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/df8c6011-efd2-4c31-9a5c-af773c0bddc4" />


**Decision support loop:** clinical question and patient context → retrieve EHR/FHIR data (demographics, notes, labs, medications, vitals, imaging) → model inference → risk score with explanation (e.g., 28% 30-day readmission risk, with factors like recent hospitalization, elevated creatinine, heart failure history, age) → clinician **accepts or overrides (with reason noted)** → outcome monitoring (accuracy, calibration, fairness, safety, clinical impact) → model refinement, with updates controlled and revalidated.

**Key points**
- Inputs: age, diagnoses, medications, labs, vitals, prior utilization, text-derived features. Outputs: probability or risk category (readmission, deterioration, sepsis, adverse events, triage priority).
- The model learns **statistical associations** from historical data. It doesn't understand medicine like a clinician, and association is not causation.
- A 20% readmission risk doesn't mean "discharge or not." The useful question is how it should guide follow-up, resources and care given the clinical context.

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/aea94980-3a6c-4f54-ab0a-850d0de063bc" />

**Calibration: probability must mean what it says**
- **Discrimination ≠ calibration.** A model can rank patients correctly yet give misleading probabilities.
- Well-calibrated means that among patients given ~30% risk, about 30% actually experience the event.
- Thresholds, bed allocation, follow-up intensity and alarm rules depend on these values. Too high → over-reaction; too low → missed patients.
- Check calibration on the **target population** and monitor it **after deployment**.

**Explanations for EHR predictions**
- Show key contributing factors (recent admissions, lab trends, oxygen requirement).
- Good: *"Risk elevated because of recent admission, rising creatinine, and oxygen requirement."* Bad: *"Model says high risk."*
- Explanations describe model behavior; they don't prove a feature *caused* the outcome.
- Show output as a draft signal, allow override, and capture the reason. Clinician action stays visible and auditable.

**Alert fatigue: a systems problem**
- A statistically strong model that fires hundreds of low-value alerts gets ignored.
- Choose thresholds based on workflow, resources and harm-benefit tradeoffs: who gets the alert, is the timing right, are suggested actions feasible, what does a false alert cost?
- AUROC alone isn't enough. Useful alerts are **specific, timely, evidence-linked and actionable**; harmful ones are frequent, vague, late, uncalibrated or impossible to act on.
- **Workflow design is part of model performance.**

## What's next

The next session covers **clinical NLP, RAG and controlled agents**, followed by privacy, governance, validation, monitoring, and the synthetic-case demo (evidence-grounded clinical note).

## Quick self-check questions

1. Why is a feedback loop essential in a deployed healthcare AI system?
2. Why can a model with high AUROC still be unsafe to deploy?
3. Give two reasons a heatmap can't be treated as proof of correctness.
4. What's the difference between discrimination and calibration?

<img width="1055" height="1491" alt="image" src="https://github.com/user-attachments/assets/2b2bb4e8-3295-478f-a7bc-67ca2d5477f1" />

