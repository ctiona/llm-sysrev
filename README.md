# LLM-Based Systematic Review Screening

This repository implements three LLM-agent frameworks for automating the **screening step** of a systematic review (title/abstract and full-text screening). 

---

## Frameworks

| ID | Description |
|----|-------------|
| **B1** | Single basic agent. Title/abstract screening followed by forced re-screening of uncertain decisions. |
| **B2-CR** | Single basic agent called twice as two independent reviewers. Conflicts are resolved through a multi-turn conversation. |
| **Pn-CR** | Pool of agents with distinct expert backgrounds. Two agents are drawn at random per paper; conflicts are resolved through a multi-turn conversation. Any number of agents ≥ 2 can be defined. |

The framework names reflect the design used in the original study (B1 = 1 basic agent, B2-CR = 1 basic agent with conflict resolution, Pn-CR = pool of n persona agents with conflict resolution). Pool size for Pn-CR is freely configurable via the `agents` list in the config file.

All three frameworks include: Title/abstract screening, automatic PDF retrieval (CrossRef, Unpaywall, Europe PMC, Semantic Scholar) with manual upload fallback, PDF pre-processing (token-budget limiting), and full-text screening.

---

## Repository structure

```
llm-systematic-review/
├── agent_setup.py           # LLM client and Agent class
├── utils.py                 # Shared utilities: PDF, XML, tokenisation
├── syscontext.py            # System-prompt template, fill in for your own review
├── requirements.txt
├── B1/
│   ├── B1.py
│   └── config.example.yaml  # Configuration file for B1, fill in for your own review
├── B2-CR/
│   ├── B2-CR.py
│   └── config.example.yaml  # Configuration file for B2-CR, fill in for your own review
└── Pn-CR/
    ├── Pn-CR.py
    └── config.example.yaml  # Configuration file for Pn-CR, fill in for your own review
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.10 or later is required.

### 2. Configure credentials

The code works with any **OpenAI-compatible** API endpoint. Set two environment variables before running:

```bash
# Linux / macOS
export OPENAI_BASE_URL=https://your-endpoint/api
export OPENAI_API_KEY=your-key-here
```

```powershell
# Windows (PowerShell)
$env:OPENAI_BASE_URL = "https://your-endpoint/api"
$env:OPENAI_API_KEY  = "your-key-here"
```

### 3. Adapt `syscontext.py`

**This is the most important setup step.** The file ships with placeholder content; you must fill it in before running any framework. Open it and replace:

| Section | What to fill in |
|---------|----------------|
| `Background / "Basic"` | Your review title. Used by B1 and B2-CR. |
| `Background / "Reviewer_1"` … | One background per Pn-CR agent: role, domain expertise, brief biography. Used by Pn-CR only. Each key must match an agent name in `Pn-CR/config.yaml`. |
| `Criteria / "t&a_Include/Exclude/Unsure"` | Your T&A INCLUDE / EXCLUDE / UNSURE rules. |
| `Criteria / "t&a_Include/Exclude"` | Same rules without UNSURE (used in B1 forced re-screening). |
| `Criteria / "fulltext_Include/Exclude"` | Your full-text INCLUDE / EXCLUDE rules. |

The `Task`, `Output_format`, and `Validity_Check` sections work for any standard systematic review and do not need to be changed.

---

## Running a framework

```bash
python B1/B1.py --config B1/config.yaml
python B2-CR/B2-CR.py --config B2-CR/config.yaml
python Pn-CR/Pn-CR.py --config Pn-CR/config.yaml
```

Each framework runs `n_cycles` independent iterations and writes one XML output file per run. A summary JSON is written at the end.

---

## Input format

The framework expects an **XML reference file** with the following structure per entry:

```xml
<root>
  <entry>
    <id>1234</id>
    <title>...</title>
    <abstract>...</abstract>
    <authors>...</authors>
    <journal>...</journal>
    <year>2023</year>
    <doi>10.xxxx/xxxxxx</doi>
  </entry>
  ...
</root>
```

This format can be exported from most reference managers (e.g. EndNote, Zotero with a custom export template), or created from a CSV file.

---

## PDF download

The framework attempts to retrieve PDFs automatically from four open-access sources in order: **CrossRef**, **Unpaywall**, **Europe PMC**, and **Semantic Scholar**. Papers that cannot be downloaded automatically are listed; you can place them manually in `manual_folder` and the framework will copy them on the next run.

To permanently skip papers that are known to be unavailable, add their IDs to `skip_ids` in the config.

---

## Generation parameters

For reasoning models that benefit from explicit sampling settings, add a `generation_kwargs` block to your config:

```yaml
generation_kwargs:
  temperature: 0.6
  top_p: 0.95
```

Omit this block for models that do not require it.

---
