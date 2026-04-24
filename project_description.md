# ReliefFlow: AI Case Management for Humanitarian Aid Organizations

## When a Spreadsheet Is the Difference Between Help and Despair

Picture a charity coordinator on a Monday morning. She is responsible for 800 families across three cities. Her records live in six different Excel files — one per volunteer, formatted differently, some in Arabic, some mixed. A field worker called on Friday about a widow with five children, one of whom needs dialysis. The note is somewhere in a WhatsApp message. By the time she finds it, triages it, and figures out what the family actually needs, three days have passed.

This is the daily reality for hundreds of humanitarian organizations working in displacement corridors across Syria, Lebanon, Turkey, and beyond. The families they serve have survived war, displacement, and loss. The organizations trying to help them are drowning in unstructured data and doing triage with tools designed for business operations.

ReliefFlow is my attempt to give those coordinators something better — a local, AI-powered case management system built around Gemma4, designed to work in the exact conditions these organizations operate in.

---

## The Real Problem Is Not Data. It Is Structure.

These organizations are not short on information. Field workers visit families constantly. Forms are filled out. Notes are taken. Voice messages are sent. The problem is that none of it is structured in a way that allows a coordinator to answer the most basic operational question: **who needs the most help right now, and what specifically do they need?**

Answering that question by reading 800 individual records is not feasible. So coordinators rely on memory and gut feeling. The families with an advocate get prioritized. The most isolated and vulnerable — the ones who have no one to speak up for them — fall to the bottom.

ReliefFlow is built to make vulnerability systematic rather than social.

---

## What ReliefFlow Does

### Ingesting the Chaos

The first thing the application does is accept data in whatever form it already exists. Organizations upload their existing Excel files — header rows in different positions, Arabic column names, merged cells, inconsistent formatting across sheets. An auto-detection engine scans each file for Arabic header keywords, reconstructs the schema, maps everything to a unified structure, and produces a clean working dataset.

No data migration project. No consultant required. You upload what you have.

### Scoring Vulnerability, Not Priority

Once the data is loaded, every family record is analyzed for vulnerability signals extracted from the free-text fields in their own language. Is there a mention of widowhood? Displacement? A medical condition? Homelessness? Unemployment? No breadwinner after a death?

These signals feed a scoring engine that assigns a priority tier — **CRITICAL**, **HIGH**, **MEDIUM**, or **LOW** — based on the combination and weight of factors. A displaced widow with a child requiring medication and no income scores differently than a family that is renting and unemployed but otherwise stable. The score reflects reality, not the order records were entered.

A coordinator can open the priority list and immediately see every CRITICAL family sorted to the top, with the specific flags that explain why.

### From Records to Procurement

Understanding individual families is necessary but not sufficient. Coordinators also need to answer operational questions across the entire caseload: how many families need food baskets this month? How many need medical supply support? How many need emergency shelter referrals?

ReliefFlow generates an aggregate needs report by mapping vulnerability signals to concrete procurement items — blankets for displaced families, hygiene kits for widows, medication support for families with chronic illness, school supplies for orphaned children. The output is a structured table with counts and urgency levels: a direct input to a procurement meeting.

---

## Gemma4 as a Field Tool, Not a Cloud Service

This is where the design philosophy of ReliefFlow diverges sharply from what most AI applications look like.

Every AI feature in ReliefFlow runs through **Gemma4, locally, via Ollama**. No API key. No data transmitted to an external server. No usage logs sitting on a cloud provider's infrastructure.

This is not a cost optimization. It is an ethical requirement.

The families in these records are among the most vulnerable people in the world. Their addresses, health conditions, family compositions, and displacement histories — this information, in the wrong hands, is dangerous. Many of them are in active conflict zones or have fled persecution. An organization that processes this data through a commercial AI API is making a trust decision on behalf of people who have no ability to consent to or contest it.

Local Gemma4 means the model runs on the coordinator's own hardware. The data never leaves the building. The organization retains complete control. And because the system runs locally, it also works in the low-connectivity environments where many of these organizations operate — field offices in camps, coordination centers in areas with unreliable internet.

---

## Multimodal Logging: Meeting Field Workers Where They Are

The most thoughtful data infrastructure fails if field workers cannot use it. And field workers in humanitarian contexts are not sitting at desks with reliable laptops. They are visiting families in informal settlements, traveling between cities, working in conditions where stopping to type structured records is simply not possible.

ReliefFlow uses Gemma4's multimodal capabilities to solve this at the point of data entry.

### Logging by Photo

A field worker photographs a handwritten Arabic intake form — the kind that has been used by these organizations for decades, filled out with a pen on a clipboard during a home visit. ReliefFlow passes that image directly to Gemma4's vision model. The model reads the handwriting, handles mixed Arabic-English text, and extracts every structured field: family size, current address, humanitarian situation, stated needs, phone number, intermediary. A fully scored record appears in the system within seconds.

The field worker never transcribes anything. The coordinator never chases down a form.

### Logging by Voice

Sometimes there is no form at all. A worker has just left a family visit and wants to capture what they learned before they forget. They open the app on their phone and record a voice memo — thirty seconds, describing the family's situation in Arabic, or English, or both.

ReliefFlow transcribes the audio using **Whisper**, a local speech recognition model that runs entirely on-device and handles Arabic natively. The transcript goes to Gemma4, which extracts the structured fields and adds a new record to the dataset, complete with priority scoring.

A spoken description becomes a structured, queryable case entry in under a minute.

### AI Insights Across the Full Caseload

Beyond individual records, Gemma4 can analyze the entire loaded dataset and surface strategic observations — which vulnerability categories are most prevalent, where resource concentration is highest, what gaps exist between what families need and what typical distributions provide. These are not generic summaries. They are grounded in the actual numbers of the loaded caseload, translated into recommendations a coordinator can act on in their next team meeting.

---

## The Design Constraint That Shaped Everything

Building this application for a hackathon around Gemma4 forced a clarifying constraint: the model had to do real work, not decorative work.

It would have been easy to use a capable local model as a thin layer on top of a rules-based system — have it reformat text that was already structured, or generate summaries of data that was already clean. That is not what ReliefFlow does.

Gemma4 is doing the hard part: reading handwriting in a language it was not specifically fine-tuned for, understanding the meaning of spoken descriptions of human suffering well enough to extract structured facts, connecting free-text situational descriptions to specific procurement categories, and answering natural-language queries against a dataset it has never seen before.

The vulnerability scoring and needs computation run on deterministic rules, which is appropriate — you do not want a probabilistic model deciding whether a family qualifies for critical-tier support. But everything involving unstructured human input — voice, image, natural language — flows through Gemma4, because that is exactly what a capable multimodal model is for.

---

## Who This Is For

ReliefFlow is designed for small to mid-size humanitarian organizations that are managing meaningful caseloads — hundreds to a few thousand families — without dedicated data infrastructure or technical staff. The kind of organization that is doing important work on a budget, where the coordinator is also the accountant and sometimes the driver.

The application requires no database setup, no cloud account, no technical configuration beyond installing Ollama and pulling the model. The interface is in English, but the system processes Arabic data natively. It works with the Excel files organizations already have.

The goal is not to replace the human judgment of experienced coordinators. It is to make sure that judgment is informed by the full picture — not just the loudest cases, not just the most recent records, not just the families someone happens to remember.

Every family in the dataset has a name. ReliefFlow's job is to make sure none of them are invisible.
