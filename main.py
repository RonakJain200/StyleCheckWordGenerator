import pandas as pd
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["http://127.0.0.1:5500", "https://yourwebsite.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Hugging Face API details
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_API_KEY = "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"  # Replace with your actual token
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# Map apparel types to CSV files
CSV_MAP = {
    "shirt": "data/shirts.csv",
    "pants": "data/pants.csv",
    "suit": "data/suits.csv",
    "tshirt": "data/tshirts.csv"
}

def get_attributes_from_csv(apparel_type, product_id=None):
    """Fetch attributes from CSV if not provided directly."""
    if apparel_type not in CSV_MAP:
        return None
    df = pd.read_csv(CSV_MAP[apparel_type])
    if product_id is not None:
        row = df.iloc[product_id]
    else:
        row = df.sample(1).iloc[0]
    return "\n".join([f"{col}: {row[col]}" for col in df.columns])

def query_llama(attributes_text):
    """Send attributes to LLaMA API for naming + descriptions."""
    prompt = f"""
    You are a fashion product naming and description expert.

    Based on the following clothing attributes, create:

    1. A 3-word product name
    2. A 5-word product name
    3. An 8-word product name
    4. A short description (2 sentences)
    5. A long description (detailed, include 3-4 bullet points)

    Clothing Attributes:
    {attributes_text}

    Format JSON exactly as:
    {{
      "3_word_name": "...",
      "5_word_name": "...",
      "8_word_name": "...",
      "short_description": "...",
      "long_description": ["point 1", "point 2", "point 3"]
    }}
    """

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "meta-llama/Meta-Llama-3-8B-Instruct:novita"
    }

    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    return response.json()

@app.post("/generate")
async def generate_product_details(request: Request):
    """API endpoint to receive apparel details and return names/descriptions."""
    data = await request.json()
    apparel_type = data.get("apparel_type")
    attributes = data.get("attributes")
    product_id = data.get("product_id")

    if not attributes:
        attributes = get_attributes_from_csv(apparel_type, product_id)

    if not attributes:
        return JSONResponse({"error": "Invalid apparel type or attributes"}, status_code=400)

    llama_response = query_llama(attributes)
    return JSONResponse(llama_response)
