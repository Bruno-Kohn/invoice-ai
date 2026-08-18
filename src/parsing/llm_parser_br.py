"""LLM parser adapted for Brazilian receipts (Nota Fiscal / Cupom Fiscal)."""

import json
import os
from typing import Optional

from google import genai

from src.parsing.schema_br import NotaFiscal


SYSTEM_PROMPT_BR = """Extraia os dados deste cupom fiscal brasileiro como JSON.

Chaves obrigatórias:
- "cnpj": string (ou null)
- "razao_social": string (ou null)
- "itens": lista de objetos com "descricao", "quantidade", "valor_unitario", "valor_total" (strings)
- "subtotal": string (ou null)
- "desconto": string (ou null)
- "valor_total": string com valor total
- "forma_pagamento": string (ou null)
- "valor_pago": string (ou null)
- "troco": string (ou null)
- "impostos": objeto com "valor_aproximado_tributos" (ou null)
- "cpf_consumidor": string (ou null)
- "data_emissao": string (ou null)

IMPORTANTE: todos os valores devem ser STRINGS, não números. Ex: "94,51" e não 94.51.
Retorne APENAS JSON válido."""


def parse_nota_fiscal(
    ocr_text: str,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.5-flash",
) -> NotaFiscal:
    """Parse OCR text from a Brazilian receipt into a NotaFiscal.

    Args:
        ocr_text: Raw text from OCR engine.
        api_key: Google AI API key. Falls back to GOOGLE_API_KEY env var.
        model_name: Gemini model to use.

    Returns:
        Parsed NotaFiscal object.
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Google API key required. Set GOOGLE_API_KEY env var or pass api_key parameter."
        )

    client = genai.Client(api_key=key)

    prompt = f"{SYSTEM_PROMPT_BR}\n\nTexto OCR do cupom fiscal:\n\n{ocr_text}"

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )

    try:
        parsed = json.loads(response.text)
    except json.JSONDecodeError:
        import re
        text = response.text
        text = re.sub(r',\s*([}\]])', r'\1', text)
        parsed = json.loads(text)

    # Convert any numeric values to strings for Pydantic compatibility
    def stringify(obj):
        if isinstance(obj, dict):
            return {k: stringify(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [stringify(v) for v in obj]
        elif isinstance(obj, (int, float)):
            return str(obj)
        return obj

    parsed = stringify(parsed)
    return NotaFiscal(**parsed)
