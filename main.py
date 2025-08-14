import os
import json
import re
import random
import requests
import pandas as pd
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct:novita")
API_URL = "https://router.huggingface.co/v1/chat/completions"

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set. Add it in your Codespace as an env var or .env file.")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Map apparel → CSV path
CSV_FILES = {
    "shirt": "data/shirts.csv",
    "suit": "data/suits.csv",
    "pant": "data/pants.csv",
    "pants": "data/pants.csv",
    "tshirt": "data/tshirts.csv",
    "t-shirt": "data/tshirts.csv"
}

app = FastAPI(title="Fashion Generator API", version="1.0.0")

# CORS (limit in production)
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    apparel_type: str
    filters: Optional[Dict[str, Any]] = None  # optional: narrow down by specific columns

def _normalize_apparel(apparel: str) -> str:
    a = apparel.strip().lower()
    if a in ("pants", "pant"):
        return "pant"
    if a in ("tshirt", "t-shirt"):
        return "tshirt"
    return a

def load_random_product(apparel_type: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    key = _normalize_apparel(apparel_type)
    if key not in CSV_FILES:
        raise HTTPException(status_code=400, detail=f"Invalid apparel_type. Choose one of: {sorted(set(CSV_FILES.keys()))}")

    path = CSV_FILES[key]
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail=f"CSV not found at {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise HTTPException(status_code=404, detail="CSV is empty.")

    # Apply simple equality filters if provided
    if filters:
        for col, val in filters.items():
            if col in df.columns:
                df = df[df[col].astype(str).str.lower() == str(val).lower()]

        if df.empty:
            raise HTTPException(status_code=404, detail="No rows matched the given filters.")

    row = df.sample(1).fillna("").iloc[0]
    return {col: row[col] for col in df.columns}

def sanitize_and_parse_json(s: str) -> Dict[str, Any]:
    # Remove code fences if the model adds them
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE)
    # Grab the first {...} block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    cleaned = m.group(0)
    return json.loads(cleaned)

def build_prompt(attributes: Dict[str, Any]) -> str:
    attributes_text = "\n".join(f"{k}: {v}" for k, v in attributes.items())
    return f"""
You are a fashion product naming and description expert.

Based on the following clothing attributes, create:

1. A 3-word product name
2. A 5-word product name
3. An 8-word product name
4. A short description (2 sentences)
5. A long description (detailed, include 3-4 bullet points)

Clothing Attributes:
{attributes_text}

Format your answer strictly as valid JSON:
{{
  "three_word_name": "...",
  "five_word_name": "...",
  "eight_word_name": "...",
  "short_description": "...",
  "long_description": [
    "point 1",
    "point 2",
    "point 3"
  ]
}}
""".strip()

def call_hf_chat(prompt: str) -> Dict[str, Any]:
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 380,
        "stream": False
    }
    r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    content = r.json()["choices"][0]["message"]["content"]
    try:
        return sanitize_and_parse_json(content)
    except Exception as e:
        # Return raw content to help diagnose if formatting slips
        raise HTTPException(status_code=502, detail=f"Invalid JSON from model: {e}\nRaw: {content}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: GenerateRequest):
    attrs = load_random_product(req.apparel_type, req.filters)
    prompt = build_prompt(attrs)
    generated = call_hf_chat(prompt)
    return {"attributes": attrs, "generated": generated}
