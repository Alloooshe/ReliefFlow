# ReliefFlow

**AI-powered humanitarian aid management for charities working at scale.**

ReliefFlow helps humanitarian organizations cut through the chaos of managing hundreds — or thousands — of vulnerable families. It turns fragmented handwritten records, audio field notes, and Excel spreadsheets into a structured, searchable, AI-analyzed case management system — all running locally, with no data ever leaving the organization's hands.

---

## The Problem

Humanitarian organizations operating in conflict zones and displacement crises face a brutal coordination challenge. A single charity might be responsible for thousands of families scattered across multiple cities, each with a different profile of need: a widow with four children and no income, a displaced family of eight living in a rented room with a child requiring ongoing medical treatment, an elderly couple with no breadwinner after losing their son.

The records for these families often exist as handwritten Arabic forms, voice messages from field workers, or inconsistently formatted Excel files — one per district, one per volunteer, one per month. There is no standard. There is no system. Workers spend hours a week just trying to figure out who needs what most urgently, and critical cases fall through the cracks.

The result: aid is distributed by familiarity rather than need. The most vulnerable families — those without a vocal advocate — are systematically under-served.

ReliefFlow is built to fix that.

---

## What It Does

### Priority Scoring — Who Needs Help First

Every family record is automatically scored across vulnerability dimensions: widowhood, displacement, homelessness, unemployment, medical conditions, disability, pregnancy, orphaned children. A point-based engine assigns a priority tier — **CRITICAL**, **HIGH**, **MEDIUM**, or **LOW** — so coordinators can triage hundreds of families in seconds instead of hours.

### Aggregate Needs Report — What to Procure

Rather than reading every record individually, coordinators can generate a consolidated procurement view: how many families need monthly food baskets, how many need medication support, how many need emergency shelter. Quantities are derived automatically from the family vulnerability signals, giving procurement teams concrete numbers to act on.

### AI Insights — Strategic Overview

Gemma4 analyzes the full caseload and surfaces actionable recommendations: where to concentrate resources, which vulnerability categories are overrepresented, what gaps exist between stated needs and current coverage.

### Smart Search — Natural Language Queries

Field workers and coordinators can query the dataset in plain English: *"Show me all displaced families with medical conditions in Aleppo"* or *"List families with more than 6 members and no income."* Gemma4 converts the question into a live filter — no SQL, no spreadsheet formulas.

### Family Profiles — Individual Case View

Each family gets a structured profile with their priority score, vulnerability flags, a generated needs list, and an AI-written case summary in English — even when the underlying record is in Arabic.

---

## Privacy-First by Design: Local Gemma4

Every AI operation in ReliefFlow runs through **Gemma4 via Ollama**, a locally hosted model. This is a deliberate architectural choice, not a convenience.

Humanitarian data is among the most sensitive that exists. Family names, addresses, phone numbers, health conditions, and displacement histories — if leaked, this information can endanger lives. Many of the families in these records are in active conflict zones or have fled persecution.

Because Gemma4 runs locally:

- **No family record, image, or voice recording is ever sent to an external API.**
- No cloud AI provider sees the data. No logs are stored on third-party servers.
- The organization retains complete control over who can access what.
- The system works in low-connectivity environments — field offices, refugee camps, areas with unreliable internet.

For deployments on Streamlit Cloud, the application connects to a self-hosted Ollama instance via a private tunnel (Cloudflare Tunnel or ngrok), keeping all inference on the organization's own hardware while still providing browser-based access to field staff.

---

## Multimodal Data Entry: Logging Records in the Field

Field workers rarely have time to sit at a laptop. ReliefFlow uses Gemma4's multimodal capabilities to meet them where they are.

### Image Parsing — Handwritten Forms

A field worker photographs a handwritten Arabic intake form with their phone. ReliefFlow uploads the image directly to Gemma4's vision model, which reads the handwriting and extracts all structured fields: family size, address, situation description, stated needs, phone number, intermediary name. The parsed record is immediately added to the dataset, scored, and available for coordinators.

No manual transcription. No data entry backlog.

### Voice Recording — Spoken Case Notes

When a field worker has just met a family and wants to log the encounter, they record a voice memo directly in the browser. ReliefFlow transcribes the audio using **Whisper** (running locally, ~145 MB model, supports Arabic and mixed-language speech), then passes the transcript to Gemma4 to extract the structured record fields. The result is the same structured entry as any other method.

A worker can describe a family's situation in 30 seconds of speech and have a fully scored, searchable record in the system within a minute.

---

## Workflow

```
Field worker photographs form  →  Gemma4 Vision parses it
            or
Field worker records voice note →  Whisper transcribes → Gemma4 structures it
            or
Coordinator uploads Excel file  →  Auto-ingested, headers auto-detected

                        ↓

              Vulnerability signals extracted
              Priority score assigned (CRITICAL / HIGH / MEDIUM / LOW)
              Needs list generated

                        ↓

          Dashboard: Priority list · Needs report · AI insights · Smart search
```

---

## Tech Stack

| Component | Technology |
|---|---|
| AI / LLM | Gemma4 (`gemma4:latest`) via Ollama |
| Speech-to-text | faster-whisper (`base` model, local) |
| UI | Streamlit |
| Data | pandas, openpyxl |
| Charts | Plotly |
| Language support | Arabic, English, mixed |
| Deployment | Streamlit Cloud + Cloudflare Tunnel (Ollama stays local) |

---

## Quickstart

### Prerequisites

- [Ollama](https://ollama.com/) installed and running
- Gemma4 pulled: `ollama pull gemma4:latest`
- Python 3.11+

### Install

```bash
git clone https://github.com/Alloooshe/ReliefFlow.git
cd ReliefFlow
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`. Load the anonymized sample dataset from the sidebar to explore all features.

### Deploy on Streamlit Cloud

1. Push the repo to GitHub (secrets are gitignored — never committed).
2. Start a Cloudflare Tunnel on the machine running Ollama:
   ```bash
   cloudflared tunnel --url http://localhost:11434
   ```
3. Copy the tunnel URL (e.g. `https://xxxx.trycloudflare.com`).
4. In Streamlit Cloud → app Settings → Secrets, add:
   ```toml
   OLLAMA_HOST = "https://xxxx.trycloudflare.com"
   ```
5. Reboot the app. The sidebar will confirm the remote connection.

---

## Data Privacy Notes

- The included sample file (`data_anonymized.xlsx`) has all names, phone numbers, intermediary names, and donor references removed.
- Raw files containing PII are gitignored and must never be committed.
- All AI inference runs on the operator's own hardware — no data is sent to Anthropic, Google, OpenAI, or any external service.

---

## Built For

The Gemma4 Kaggle Hackathon — demonstrating real-world humanitarian applications of local multimodal AI.

The application is designed around actual workflows used by charities operating in Syria, Lebanon, and neighboring displacement corridors, where the combination of Arabic-language records, limited connectivity, and extreme sensitivity of beneficiary data makes locally-run AI the only responsible choice.
