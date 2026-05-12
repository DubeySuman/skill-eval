# Skill Eval — Groq Edition

Test any Claude Agent Skill for free using the Groq API (LLaMA 3.3 70B).

## Quick Start

```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Open: http://localhost:8000

## How to use

1. Enter your Groq API key (`gsk_...`) from console.groq.com
2. Paste your SKILL.md content — tests are auto-generated
3. Click a test card to run it, or write your own in the Custom Prompt area
4. Use Batch tab to run all tests at once for regression testing

## Deploy to Railway

1. Push this folder to a public GitHub repo
2. Go to railway.app → New Project → Deploy from GitHub
3. Select the repo → wait ~2 min
4. Settings → Networking → Generate Domain

No environment variables needed. Users supply their own Groq key.
