"""Streamlit app for Invoice AI — Upload a receipt image and get structured JSON."""

import streamlit as st
import cv2
import numpy as np
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.document_detector import detect_document
from src.preprocessing.deskew import deskew
from src.preprocessing.enhancement import enhance
from src.preprocessing.transforms import apply_bilateral_filter
from src.ocr.postprocessing import correct_numeric_chars, normalize_whitespace, merge_lines_by_proximity


st.set_page_config(
    page_title="Invoice AI",
    page_icon="🧾",
    layout="wide",
)

st.title("🧾 Invoice AI")
st.markdown("**Upload a receipt image → Get structured JSON**")
st.markdown("---")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    parser_choice = st.selectbox("Parser", ["LLM (Gemini 3.5 Flash)", "Regex"])
    language = st.selectbox("Language", ["Português (BR)", "English / International"])
    preprocessing = st.checkbox("Apply preprocessing", value=True)
    show_ocr = st.checkbox("Show OCR text", value=True)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    Pipeline: Image → Preprocessing → OCR → Parser → JSON
    
    - **OCR**: PaddleOCR v6
    - **LLM**: Google Gemini 3.5 Flash
    - **CNN**: MobileNetV2 Quality Gate
    """)

# Main area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Upload Receipt")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
    )

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display original
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)

with col2:
    st.subheader("📄 Extracted JSON")

    if uploaded_file is not None:
        with st.spinner("Processing..."):
            # Preprocessing
            if preprocessing:
                processed = detect_document(image)
                processed = deskew(processed)
                processed = apply_bilateral_filter(processed)
                processed = enhance(processed, grayscale=True, clahe=True, adaptive_threshold=False)
                if len(processed.shape) == 2:
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            else:
                processed = image

            # OCR
            from src.ocr.paddle_ocr import PaddleOCREngine
            
            @st.cache_resource
            def get_ocr_engine():
                return PaddleOCREngine(language="en")

            engine = get_ocr_engine()
            results = engine.recognize(processed)
            merged = merge_lines_by_proximity(results)
            lines = [normalize_whitespace(correct_numeric_chars(r.text)) for r in merged if r.text.strip()]
            ocr_text = "\n".join(lines)

            if show_ocr:
                with st.expander("🔍 OCR Text", expanded=False):
                    st.code(ocr_text)

            # Parsing
            try:
                if parser_choice == "LLM (Gemini 3.5 Flash)":
                    api_key = os.environ.get("GOOGLE_API_KEY")
                    if not api_key:
                        st.error("⚠️ Set GOOGLE_API_KEY environment variable to use LLM parser")
                        st.stop()

                    if language == "Português (BR)":
                        from src.parsing.llm_parser_br import parse_nota_fiscal
                        receipt = parse_nota_fiscal(ocr_text, api_key=api_key)
                    else:
                        from src.parsing.llm_parser import parse_receipt
                        receipt = parse_receipt(ocr_text, api_key=api_key)
                else:
                    from src.parsing.regex_parser import parse_receipt
                    receipt = parse_receipt(ocr_text)

                # Display JSON
                result_json = receipt.model_dump()
                st.json(result_json)

                # Download button
                json_str = json.dumps(result_json, indent=2, ensure_ascii=False)
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_str,
                    file_name="receipt.json",
                    mime="application/json",
                )

                # Metrics
                if hasattr(receipt, 'valor_total') and receipt.valor_total:
                    st.success(f"✅ Total: R$ {receipt.valor_total}")
                elif hasattr(receipt, 'total') and receipt.total and receipt.total.total_price:
                    st.success(f"✅ Total: {receipt.total.total_price}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    else:
        st.info("👈 Upload a receipt image to get started")

# Footer
st.markdown("---")
st.markdown("Built with PaddleOCR, Gemini 3.5 Flash, PyTorch | [GitHub](https://github.com/Bruno-Kohn/invoice-ai)")
