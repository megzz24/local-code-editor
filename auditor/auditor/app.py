import json
import tempfile

import streamlit as st

from services.paddle_service import PaddleOCRService
from services.receipt_parser import ReceiptParser

@st.cache_resource
def get_ocr_service():
    return PaddleOCRService()


@st.cache_resource
def get_receipt_parser():
    return ReceiptParser()


@st.cache_resource
def get_finetuned_auditor():
    from services.auditor.inference import ExpenseAuditorInference
    return ExpenseAuditorInference()

st.set_page_config(page_title="Policy-First Expense Auditor", page_icon="🧾")
st.title("🧾 Policy-First Expense Auditor")
st.caption("Upload a receipt, fill in a few details, and get an instant policy-based verdict.")

st.header("1. Upload Receipt")
uploaded_file = st.file_uploader(
    "Upload a receipt image (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded receipt", use_column_width=True)

st.header("2. Claim Details")

col1, col2 = st.columns(2)

with col1:
    employee_level = st.selectbox(
        "Employee Level",
        options=["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L7+"],
        index=2,
        help="Used to determine your seniority tier (Junior/Mid-Level/Senior/Executive).",
    )

with col2:
    location = st.text_input(
        "Location (city)",
        value="Bengaluru",
        help="Used to apply the regional cost-of-living multiplier (Section 11).",
    )

business_purpose = st.text_input(
    "Business Purpose",
    value="",
    placeholder="e.g. Client dinner with vendor, Team lunch, Taxi to airport",
    help="Used to match this expense against a spending category in Section 3.",
)

st.header("3. Run Audit")

run_button = st.button("Run Audit", type="primary")

if run_button:
    if uploaded_file is None:
        st.error("Please upload a receipt image first.")
        st.stop()

    if not business_purpose.strip():
        st.error("Please enter a business purpose.")
        st.stop()

    with tempfile.NamedTemporaryFile(
        delete=False, suffix="." + uploaded_file.name.split(".")[-1]
    ) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_image_path = tmp_file.name

    with st.spinner("Running OCR on the receipt..."):
        ocr_service = get_ocr_service()
        ocr_text = ocr_service.extract_text(tmp_image_path)

    with st.expander("📄 Raw OCR Text (for debugging)"):
        st.text(ocr_text if ocr_text else "(no text detected)")

    if not ocr_text.strip():
        st.error("OCR could not read any text from this image. Try a clearer photo.")
        st.stop()

    with st.spinner("Parsing receipt with local LLM..."):
        parser = get_receipt_parser()
        try:
            receipt_data = parser.parse(ocr_text)
        except Exception as e:
            st.error(f"Failed to parse receipt: {e}")
            st.stop()

    with st.expander("🧾 Parsed Receipt JSON"):
        st.json(receipt_data)

    # --- Step D: Run the audit engine ---
    audit_request = {
        "employee_level": employee_level,
        "location": location,
        "business_purpose": business_purpose,
        "receipt": receipt_data,
    }

    with st.spinner("Running fine-tuned model audit..."):
        try:
            auditor = get_finetuned_auditor()
            audit_result = auditor.audit(audit_request)
        except Exception as e:
            st.error(
                f"Failed to run fine-tuned auditor: {e}\n\n"
                "Make sure outputs/lora_adapter/ exists "
                "(run training/train_lora.py first) and that you have "
                "enough free GPU memory."
            )
            st.stop()

    st.header("4. Audit Result")
    st.caption("Audit engine: Fine-tuned LLM (LoRA adapter)")

    verdict = audit_result.get("verdict", "Unknown")
    policy_section = audit_result.get("policy_section", "")
    reason = audit_result.get("reason", "")

    if verdict == "Approved":
        st.success(f"✅ Verdict: **{verdict}**")
    elif verdict == "Flagged":
        st.warning(f"⚠️ Verdict: **{verdict}**")
    else:
        st.error(f"❌ Verdict: **{verdict}**")

    st.markdown(f"**Policy Section:** {policy_section}")
    st.markdown(f"**Reason:** {reason}")

    with st.expander("🔍 Full Audit Result (JSON)"):
        st.json(audit_result)