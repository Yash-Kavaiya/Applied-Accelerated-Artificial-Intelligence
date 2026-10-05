# Week 12: Healthcare AI Applications (Part 2 Summary)

Part 1 covered the system view, medical imaging, and EHR risk prediction. This part covers clinical NLP, RAG, agents, governance, evaluation, and the demo.

## Part 3: Clinical NLP, RAG, and agents

**Core principle:** a fluent answer is not enough. Output must be grounded in patient facts and trusted evidence.

**Clinical NLP tasks**

| Task | Input | Output |
|---|---|---|
| Information extraction | notes, reports | problems, medications, symptoms, dates |
| Summarization | long chart, discharge notes | brief clinical summary |
| Report drafting | measurements, findings | draft radiology/pathology/discharge text |
| Question answering | patient chart + guidelines | answer with sources and uncertainty |
| Coding assistance | clinical documentation | suggested billing/quality codes for human review |

Each task carries different risks (a summary can omit something harmful; QA can hallucinate a recommendation), so each needs its own evaluation and review.

**Why naive clinical LLM use is unsafe**
- LLMs produce confident, fluent text even when unsupported. Fluency is not factual support.
- Failure modes: hallucinated diagnosis, unsupported treatment suggestion, missed contraindication, outdated guideline, PHI leakage, and mixing facts from different parts of the record.
- Sending identifiable patient data to unauthorized third-party APIs is a privacy breach.
- **Safe pattern:** retrieve evidence → draft answer → cite sources → check guardrails → clinician reviews.

**Clinical RAG**
- **Flow:** clinician question → retriever searches trusted guidelines → evidence snippets with citations → LLM drafts using the evidence → safety checker → clinician final review, with a feedback loop for follow-ups.
- Versus a plain LLM, which answers from training memory (possibly outdated, no visible evidence), RAG uses retrieved passages and exposes citations. It helps especially with **local protocols**.
- **RAG does not remove hallucination.** Bad retrieval can still produce a convincing but wrong answer. Source collection, retrieval quality, prompt, and review all matter, and the clinician must be able to spot a bad retrieval.
- Example contrast: *"What antibiotics should I use?"* (generic or outdated) vs. *"Using these local CAP guideline snippets, draft initial management options and cite the evidence."*

**Clinical agents: controlled tool use, not autonomy**
Definition: a clinical agent is a **controlled orchestrator of tools, not an autonomous clinician**. It reads patient context, retrieves guidelines, calls calculators, and summarizes reports, producing a **draft with sources** that a human reviews, edits, and approves.

| Tool | Role | Guardrail |
|---|---|---|
| FHIR/EHR reader | retrieve patient facts | read-only, authorized data only |
| Guideline retriever | find evidence and local protocols | source whitelist and citations |
| Clinical calculator | compute risk scores | show inputs and formula |
| Report summarizer | draft concise summaries | no new facts beyond sources |
| Audit logger | record query, sources, outputs, review | immutable log and versioning |

**Most important design choice: capability limitation.** The agent should not autonomously diagnose, prescribe, or modify the medical record.

**Five guardrails**
1. **Read-only access** by default.
2. **Source grounding:** claims point to patient facts, guideline passages, or calculator outputs.
3. **No autonomous treatment:** diagnosis, medication, and orders require a clinician.
4. **Human approval** before a draft enters clinical documentation.
5. **Auditability:** reconstruct what the system saw, which tools it called, what it generated, who reviewed it, and the final action. This matters especially because LLMs are probabilistic.

## Part 4: Privacy, governance, and regulation

**Safeguards around healthcare AI:** data minimization, consent and authorization, de-identification, encryption and access control, and audit logs and monitoring. At the organizational level, a **governance board** reviews high-risk use cases and a **risk register** tracks privacy, security, bias, and safety risks with mitigations.

- **De-identification is not zero re-identification risk.**
- **PHI practical boundaries:** minimum necessary data, remove or transform identifiers, role-based access, encrypt at rest and in transit, log access and reviewer decisions. **Never put real patient data into an unauthorized model or tool in a classroom demo.**

**Ethical principles (WHO-style):** protect autonomy, promote safety and well-being, ensure transparency and explainability, foster responsibility and accountability, ensure inclusiveness and equity, and promote responsive and sustainable AI.

**Regulatory boundary: intended use matters.** Software intended to diagnose, treat, mitigate, or prevent disease may fall under medical-device regulation, depending on jurisdiction. Claims must be precise:
- ✅ Safe wording: *"Evidence-grounded draft for clinician review."*
- ❌ Unsafe wording: *"AI diagnoses pneumonia and recommends treatment."*

The lecture focuses on engineering principles, not legal clearance advice.

## Part 4 (cont.): Evaluation and monitoring lifecycle

Don't judge a system on training data or one internal test set. The eight stages are:
1. **Dataset split:** separate development, internal test, and external validation sets.
2. **External validation:** different site, population, time period, or device. Performance can drop even with identical code because of case mix, coding practices, scanners, or lab ranges. Report where validation data came from, not just an average.
3. **Calibration:** do predicted probabilities match observed outcomes?
4. **Fairness / subgroup analysis:** average performance can hide poor performance in smaller groups (sex, age, site, ethnicity where appropriate, device, disease stage). Ask: *"Who benefits, who is missed, and who may be harmed?"* It is a **safety check, not decoration**.
5. **Silent deployment:** the model runs in the background without affecting care, collecting predictions, available inputs, outcomes, latency, missing data, alert volume, and drift signals.
6. **Drift detection:** data drift (inputs change), label drift (outcome recording changes), performance drift (accuracy, calibration, or subgroup results decline).
7. **Incident review:** detect, root-cause analysis, and multidisciplinary review.
8. **Model update and control:** retrain, re-evaluate, approve, version, redeploy, and monitor. **No retrained model should silently replace the old one without re-evaluation and approval.**

**Metrics must match the task**

| Task | Metrics | Concern |
|---|---|---|
| Binary prediction | AUROC, AUPRC, sensitivity, specificity, PPV, NPV | threshold choice, false alerts |
| Risk scoring | calibration curve, Brier score, decision curve | probability reliability |
| Segmentation | Dice, IoU, surface distance | boundary accuracy, measurement error |
| Text summarization | factual consistency, omissions, hallucinations | does the note preserve clinical facts? |
| Workflow impact | time saved, action rate, outcomes, override rate | does it help in practice? |

## Part 5: Week 12 demo (synthetic pneumonia case)

**Slogan: "Generate drafts, not decisions."** Every AI-generated statement must trace to a **patient fact**, a **retrieved guideline snippet**, or a **calculator output**. The LLM is only one part of the system. The key output is an auditable chain from facts and evidence to a clinician-reviewed draft.

**Eight-stage workflow**
1. **Synthetic patient case:** adult with cough, fever, shortness of breath, SpO2 91%, hypertension, type 2 diabetes. No real PHI.
2. **FHIR-like JSON record:** Patient, Condition, Observation, etc. Structured data is easier to audit than a free-form prompt.
3. **Guideline retrieval:** snippets on initial evaluation, chest imaging, empiric therapy, severity, disposition. Inspect snippets *before* reading the summary. Bad retrieval: wrong disease, pediatric guideline for an adult, outdated protocol, missing source.
4. **Risk calculator:** CURB-65 with explicit, verifiable inputs (0-1 low risk, 2 moderate, 3-5 high).
5. **Evidence-grounded summary:** separates patient facts from guideline suggestions, avoids treatment certainty, and notes the need for review. Good: *"consistent with possible community-acquired pneumonia; guideline snippets support chest imaging and empiric therapy consideration, with local protocol review."* Bad: *"Patient definitely has pneumonia; start antibiotic X."*
6. **Structured note draft:** Assessment and Plan, marked AI-generated and requires clinician review, with placeholders for missing data or local protocols.
7. **Clinician review:** verify accuracy, edit, confirm orders and follow-up, approve.
8. **Audit log:** timestamped record of each stage.

**Demo files:** `case_fhir.json`, `retrieved_guidelines.json`, `risk_score.json`, `summary_with_sources.txt`, `clinical_note_draft.txt`, `audit_log.jsonl`. Run step by step with scripts like `prepare_case.py`, `retrieve_guidelines.py`, `run_calculator.py`, `draft_summary.py`, `draft_note.py`, `review_audit.py`, and keep a pre-generated backup output folder.

**Troubleshooting:** slow LLM → use cached output; irrelevant snippets → shrink the corpus or improve the query; unsupported claim → show the guardrail failure and remove it in review; missing calculator input → mark unknown, never infer silently; no network → use local cached snippets.

**Class activity:** label each summary sentence as *supported by patient facts*, *supported by a guideline snippet*, or *unsupported*. Then identify clinically important missing information, judge whether the wording is too authoritative, and decide what belongs in the audit log before approval.

**Failure modes to teach**

| Failure | Why it matters |
|---|---|
| Missing patient fact | summary looks complete but omits a contraindication |
| Wrong retrieved guideline | answer is evidence-shaped but clinically irrelevant |
| Unsupported treatment claim | model oversteps from support to decision-making |
| Bad calculator input | a transparent score becomes misleading |
| Poor calibration | probabilities no longer support thresholds |
| No audit trail | system can't be investigated after an incident |

## Extensions and final synthesis

**Extensions:** add a second case (heart failure, diabetes medication review, discharge summary); compare plain LLM vs. RAG with citations; add a local embedding retriever; run an external validation exercise on synthetic cohorts; profile latency; compare deployment architectures (hospital network, private cloud, on-device, audit storage).

**Final workflow:** patient context → structured data → trusted retrieval → model or calculator → evidence-linked draft → clinician review → audit log → monitoring.

Detection, prediction, or generation alone isn't enough. The system must make **uncertainty, evidence, reviewer responsibility, and downstream action** visible. Accelerated-AI concerns (latency, throughput, memory, privacy, monitoring, versioning) still apply.

## Quick self-check questions

1. Why doesn't RAG eliminate hallucination, and what should the interface show to help a clinician catch bad retrieval?
2. List the five guardrails for a clinical agent. Which one allows incident investigation?
3. Why is external validation needed even if the code is identical?
4. What is the difference between data drift, label drift, and performance drift?
5. Why does the demo use an explicit calculator instead of letting the LLM compute CURB-65?
6. Why is "Evidence-grounded draft for clinician review" safer wording than "AI diagnoses pneumonia"?

I can turn the two summaries into a combined quiz or flashcards if you'd like.
