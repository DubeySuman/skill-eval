from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import json
import os

app = FastAPI()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


class EvalRequest(BaseModel):
    api_key: str
    skill_content: str
    test_prompt: str


class GenerateTestsRequest(BaseModel):
    api_key: str
    skill_content: str


class BatchEvalRequest(BaseModel):
    api_key: str
    skill_content: str
    test_prompts: list[str]


async def call_groq(api_key: str, messages: list, system: str = None, temperature: float = 0.7):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
    }
    if system:
        payload["messages"] = [{"role": "system", "content": system}] + messages

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(GROQ_API_URL, headers=headers, json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()["choices"][0]["message"]["content"]


@app.post("/api/generate-tests")
async def generate_tests(req: GenerateTestsRequest):
    """Analyze skill content and generate targeted test prompts."""
    system = """You are an expert AI skill tester. Analyze the provided SKILL.md content and generate exactly 5 targeted test prompts that will thoroughly evaluate the skill.

Rules:
1. Read the skill's trigger phrases, domain, and purpose carefully
2. Generate tests that cover: obvious trigger, paraphrased trigger, edge case, negative trigger (should NOT activate skill), and a complex real-world scenario
3. Label each test with its type and complexity
4. Return ONLY valid JSON, no markdown, no explanation

Output format (strict JSON):
{
  "tests": [
    {
      "label": "Short label (2-4 words)",
      "type": "POSITIVE | NEGATIVE | EDGE",
      "complexity": "LOW | MEDIUM | HIGH",
      "prompt": "The actual test prompt the user would type"
    }
  ]
}"""

    user_msg = f"Analyze this skill and generate 5 targeted tests:\n\n{req.skill_content}"

    try:
        response = await call_groq(
            req.api_key,
            [{"role": "user", "content": user_msg}],
            system=system,
            temperature=0.4
        )
        # Strip any markdown fences if present
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        parsed = json.loads(clean.strip())
        return parsed
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse test generation response")


@app.post("/api/eval")
async def eval_skill(req: EvalRequest):
    """Run a single skill evaluation."""
    # Call 1: LLaMA acts as Claude with the skill loaded
    skill_response = await call_groq(
        req.api_key,
        [{"role": "user", "content": req.test_prompt}],
        system=req.skill_content,
        temperature=0.7
    )

    # Call 2: LLaMA judges the response
    judge_system = """You are an expert AI skill evaluator. Your job is to score how well a skill-loaded AI responded to a user prompt.

Score across 5 dimensions (0-20 each, total 100):
1. TRIGGER: Did the skill correctly activate (or correctly NOT activate) for this prompt?
2. DOMAIN: Was the correct domain/category detected and applied?
3. GATE: Were appropriate validation gates or questions asked before proceeding?
4. QUESTIONS: Were questions targeted, one-at-a-time, and relevant?
5. ASSUMPTIONS: Did the AI avoid making unsupported assumptions?

Return ONLY valid JSON, no markdown, no extra text:
{
  "scores": {
    "trigger": <0-20>,
    "domain": <0-20>,
    "gate": <0-20>,
    "questions": <0-20>,
    "assumptions": <0-20>
  },
  "total": <0-100>,
  "verdict": "PASS | FAIL | PARTIAL",
  "summary": "2-3 sentence evaluation summary",
  "issues": ["issue1", "issue2"] 
}"""

    judge_prompt = f"""Skill instructions:
---
{req.skill_content}
---

User prompt: {req.test_prompt}

AI response:
{skill_response}

Score this response."""

    judge_response = await call_groq(
        req.api_key,
        [{"role": "user", "content": judge_prompt}],
        system=judge_system,
        temperature=0.2
    )

    clean = judge_response.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]

    try:
        scores = json.loads(clean.strip())
    except json.JSONDecodeError:
        scores = {
            "scores": {"trigger": 0, "domain": 0, "gate": 0, "questions": 0, "assumptions": 0},
            "total": 0,
            "verdict": "ERROR",
            "summary": "Failed to parse judge response.",
            "issues": ["JSON parse error in judge response"]
        }

    return {
        "skill_response": skill_response,
        "evaluation": scores
    }


@app.post("/api/batch-eval")
async def batch_eval(req: BatchEvalRequest):
    """Run multiple test prompts sequentially."""
    results = []
    for prompt in req.test_prompts:
        single = EvalRequest(
            api_key=req.api_key,
            skill_content=req.skill_content,
            test_prompt=prompt
        )
        result = await eval_skill(single)
        result["prompt"] = prompt
        results.append(result)
    return {"results": results}


@app.get("/api/health")
async def health():
    return {"status": "online"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
