"""LLM-based parser using Google Gemini to extract structured receipt data.

Uses Gemini 2.0 Flash with temperature=0 and JSON mode to parse
OCR text into a Receipt schema. Supports zero-shot and few-shot modes.
"""

import json
import os
from typing import Optional

from google import genai

from src.parsing.schema import Receipt


SYSTEM_PROMPT = """You are a receipt data extraction assistant. Given raw OCR text from a receipt image, extract the structured data as JSON.

The JSON must have exactly these keys:
- "menu": list of items, each with "nm" (name), "cnt" (quantity like "1 x"), "price" (total price string)
- "sub_total": object with "subtotal_price", "tax_price", "service_price", "discount_price" (all optional strings)
- "total": object with "total_price" (required), "cashprice", "changeprice" (optional strings)

Rules:
- Keep prices as strings exactly as they appear (e.g. "75,000" not 75000)
- If a field is not found, set it to null
- "cnt" should be in format "N x" (e.g. "1 x", "2 x")
- Do NOT invent data that isn't in the text
- Return ONLY valid JSON, no markdown or explanation"""


FEW_SHOT_EXAMPLES = [
    {
        "input": """Nasi Goreng  2 x  70,000
Ice Tea  1 x  18,000
Sub Total  88,000
Tax 10%  8,800
TOTAL  96,800
Cash  100,000
Change  3,200""",
        "output": {
            "menu": [
                {"nm": "Nasi Goreng", "cnt": "2 x", "price": "70,000"},
                {"nm": "Ice Tea", "cnt": "1 x", "price": "18,000"},
            ],
            "sub_total": {
                "subtotal_price": "88,000",
                "tax_price": "8,800",
                "service_price": None,
                "discount_price": None,
            },
            "total": {
                "total_price": "96,800",
                "cashprice": "100,000",
                "changeprice": "3,200",
            },
        },
    }
]


def parse_receipt(
    ocr_text: str,
    few_shot: bool = False,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.5-flash",
) -> Receipt:
    """Parse OCR text into a Receipt using Google Gemini.

    Args:
        ocr_text: Raw text from OCR engine.
        few_shot: If True, include few-shot examples in the prompt.
        api_key: Google AI API key. Falls back to GOOGLE_API_KEY env var.
        model_name: Gemini model to use.

    Returns:
        Parsed Receipt object.
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Google API key required. Set GOOGLE_API_KEY env var or pass api_key parameter."
        )

    client = genai.Client(api_key=key)

    prompt = _build_prompt(ocr_text, few_shot)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )

    parsed = json.loads(response.text)

    # Ensure total exists
    if "total" not in parsed:
        parsed["total"] = {"total_price": None}

    return Receipt(**parsed)


def _build_prompt(ocr_text: str, few_shot: bool) -> str:
    """Build the prompt for Gemini."""
    parts = [SYSTEM_PROMPT]

    if few_shot:
        parts.append("\n--- EXAMPLES ---")
        for ex in FEW_SHOT_EXAMPLES:
            parts.append(f"\nInput:\n{ex['input']}")
            parts.append(f"\nOutput:\n{json.dumps(ex['output'], indent=2)}")
        parts.append("\n--- END EXAMPLES ---\n")

    parts.append(f"\nNow extract the receipt data from this OCR text:\n\n{ocr_text}")

    return "\n".join(parts)
