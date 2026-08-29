import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import io

from src.ml_engine import ml_engine
from src.finops_agent import finops_agent
from src.security_agent import security_agent
from src.orchestrator import orchestrator

# Page Configuration
st.set_page_config(
    page_title="FinOps & Security Compliance Agent Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.5rem; }
    .subtitle { font-size: 1.0rem; color: #64748B; margin-bottom: 1.5rem; }
    .verdict-approve { background-color: #DCFCE7; color: #166534; padding: 12px; border-radius: 8px; font-weight: bold; font-size: 1.2rem; border-left: 6px solid #22C55E; }
    .verdict-block { background-color: #FEE2E2; color: #991B1B; padding: 12px; border-radius: 8px; font-weight: bold; font-size: 1.2rem; border-left: 6px solid #EF4444; }
    .verdict-route { background-color: #FEF3C7; color: #92400E; padding: 12px; border-radius: 8px; font-weight: bold; font-size: 1.2rem; border-left: 6px solid #F59E0B; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛡️ FinOps & Security Compliance Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Autonomous Financial Ops Decisioning • NeurIPS 2022 Fraud ML • SHA-256 Cryptographic Audit Ledger</div>', unsafe_allow_html=True)

# Sidebar Information
st.sidebar.image("https://img.icons8.com/color/96/000000/shield-with-signature.png", width=64)
st.sidebar.title("FinGuard Control Panel")
st.sidebar.markdown("**System Status**: 🟢 Operational")
st.sidebar.markdown("**ML Champion**: LightGBM (PR-AUC 1.00)")
st.sidebar.markdown("**PCA Variance Target**: 95% EVR Threshold")
st.sidebar.divider()

# Core Tabs
tab_sim, tab_batch, tab_pca, tab_audit = st.tabs([
    "⚡ Live Simulator",
    "📁 Batch CSV Upload",
    "📊 PCA Variance Analytics",
    "🔐 Cryptographic Audit Ledger"
])

# ---------------------------------------------------------
# TAB 1: LIVE SIMULATOR
# ---------------------------------------------------------
with tab_sim:
    st.subheader("Interactive Event Decision Simulator")
    st.write("Test incoming transaction, application, or invoice records in real time:")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### 💼 Financial Ops Inputs")
        applicant_name = st.text_input("Applicant / Customer Name", "John Smith")
        email = st.text_input("Email Address", "johnsmith@gmail.com")
        vendor_name = st.selectbox("Vendor Name", ["Acme Corp", "TechData Inc", "Global Logistics LLC", "Apex Cloud Systems", "Unknown Supplies"])
        raw_amount = st.text_input("Invoice Amount ($)", "$12,500.00")
        po_number = st.text_input("PO Number", "PO-1001")

    with col_b:
        st.markdown("### 🤖 NeurIPS ML Inputs")
        income = st.slider("Income Quantile", 0.0, 1.0, 0.5, 0.05)
        credit_score = st.slider("Credit Risk Score", 300, 850, 720)
        velocity_6h = st.number_input("Transaction Velocity (6h Count)", min_value=1, max_value=50, value=2)
        access_hour = st.slider("Time of Access (Hour 0-23)", 0, 23, 14)

    with col_c:
        st.markdown("### 🛡️ Security & Notes")
        notes = st.text_area("Transaction Notes / Description", "Routine monthly subscription invoice payment.")
        actor_id = st.text_input("Actor ID", "USR-9042")

    with st.expander("⚙️ Advanced NeurIPS ML Parameters (Optional — All 26 Features)"):
        col_adv1, col_adv2, col_adv3 = st.columns(3)
        with col_adv1:
            customer_age = st.slider("Customer Age", 18, 90, 35)
            prev_address_months = st.number_input("Prev Address Months", 0, 200, 24)
            current_address_months = st.number_input("Current Address Months", 0, 200, 36)
            bank_months_count = st.number_input("Bank Months Count", 0, 100, 12)
            has_other_cards = st.selectbox("Has Other Cards", [0, 1], index=0)

        with col_adv2:
            proposed_credit_limit = st.number_input("Proposed Credit Limit ($)", 200.0, 50000.0, 2000.0)
            intended_balcon_amount = st.number_input("Intended Transfer Amount ($)", 0.0, 50000.0, 0.0)
            bank_branch_count_8w = st.number_input("Bank Branch Count (8w)", 0, 50, 2)
            date_of_birth_distinct_emails = st.number_input("DOB Distinct Emails (4w)", 1, 20, 1)
            session_length_in_minutes = st.number_input("Session Length (Minutes)", 0.0, 120.0, 15.0)

        with col_adv3:
            device_distinct_emails_8w = st.number_input("Device Distinct Emails (8w)", 1, 10, 1)
            device_fraud_count = st.number_input("Device Fraud Count", 0, 10, 0)
            foreign_request = st.selectbox("Foreign Request", [0, 1], index=0)
            keep_alive_session = st.selectbox("Keep Alive Session", [0, 1], index=1)
            days_since_request = st.number_input("Days Since Request", 0.0, 30.0, 0.5)

    if st.button("⚡ Evaluate Decision", type="primary", use_container_width=True):
        input_data = {
            "applicant_name": applicant_name,
            "email": email,
            "vendor_name": vendor_name,
            "invoice_amount": raw_amount,
            "po_number": po_number,
            "income": income,
            "credit_risk_score": credit_score,
            "velocity_6h": velocity_6h,
            "access_hour": access_hour,
            "notes": notes,
            "actor_id": actor_id,
            "customer_age": customer_age,
            "prev_address_months_count": prev_address_months,
            "current_address_months_count": current_address_months,
            "bank_months_count": bank_months_count,
            "has_other_cards": has_other_cards,
            "proposed_credit_limit": proposed_credit_limit,
            "intended_balcon_amount": intended_balcon_amount,
            "bank_branch_count_8w": bank_branch_count_8w,
            "date_of_birth_distinct_emails_4w": date_of_birth_distinct_emails,
            "session_length_in_minutes": session_length_in_minutes,
            "device_distinct_emails_8w": device_distinct_emails_8w,
            "device_fraud_count": device_fraud_count,
            "foreign_request": foreign_request,
            "keep_alive_session": keep_alive_session,
            "days_since_request": days_since_request
        }

        res = orchestrator.process_event(input_data)
        verdict = res["final_verdict"]

        st.divider()
        st.markdown("### 🎯 Decision Verdict Output")

        if verdict == "AUTO_APPROVE":
            st.markdown('<div class="verdict-approve">🟢 AUTO_APPROVE — Straight-Through Processing Approved</div>', unsafe_allow_html=True)
        elif verdict == "AUTO_BLOCK":
            st.markdown('<div class="verdict-block">🔴 AUTO_BLOCK — Hard Risk Violation Detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict-route">🟡 ROUTE_TO_HUMAN_REVIEW — High-Impact / Ambiguous Exception Enqueued</div>', unsafe_allow_html=True)

        st.write("")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ML Fraud Score", f"{res['layer_breakdown']['ml_engine']['fraud_probability']*100:.1f}%")
        col2.metric("Extraction Confidence", f"{res['layer_breakdown']['finops_agent']['extraction_confidence']*100:.0f}%")
        col3.metric("UEBA Anomaly Score", f"{res['layer_breakdown']['security_agent']['ueba_score']}")
        col4.metric("Risk Level", res["risk_level"])

        st.markdown("#### 🔍 Evidence Rationales & Flags")
        if res["evidence_summary"]:
            for r in res["evidence_summary"]:
                st.warning(f"• {r}")
        else:
            st.success("• All FinOps and SecOps checks passed cleanly without policy findings.")

        st.markdown(f"**Cryptographic Hash**: `{res['audit_hash']}`")

# ---------------------------------------------------------
# TAB 2: BATCH CSV UPLOAD
# ---------------------------------------------------------
with tab_batch:
    st.subheader("Batch CSV Ingestion & Decision Processor")

    # Download Template Button
    template_data = "applicant_name,email,vendor_name,invoice_amount,po_number,income,velocity_6h,credit_risk_score,access_hour,notes\n" \
                    "John Smith,jsmith@gmail.com,Acme Corp,\"$4,500.00\",PO-1001,0.6,1,740,14,Monthly office supplies\n" \
                    "Alice Brown,abrown@gmail.com,Unknown Supplies,\"$14,500.00\",N/A,0.2,12,520,3,Urgent wire request\n" \
                    "David Miller,dmiller@gmail.com,TechData Inc,890.00,PO-1002,0.8,2,790,11,Cloud infrastructure\n"

    st.download_button(
        label="📥 Download Sample CSV Template",
        data=template_data,
        file_name="sample_finguard_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload CSV File for Batch Ingestion", type=["csv"])
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(df_batch)}** records from CSV:")
        st.dataframe(df_batch.head())

        if st.button("🚀 Process Batch Decisions"):
            results = []
            for _, row in df_batch.iterrows():
                row_dict = row.to_dict()
                res = orchestrator.process_event(row_dict)
                results.append({
                    "event_id": res["event_id"],
                    "vendor": row_dict.get("vendor_name") or row_dict.get("vendor"),
                    "amount": res["layer_breakdown"]["finops_agent"]["sanitized_amount"],
                    "fraud_score": res["layer_breakdown"]["ml_engine"]["fraud_probability"],
                    "verdict": res["final_verdict"],
                    "audit_hash": res["audit_hash"][:16] + "..."
                })
            
            df_res = pd.DataFrame(results)
            st.success("Batch processing complete!")
            st.dataframe(df_res)

# ---------------------------------------------------------
# TAB 3: PCA & MLFLOW EXPERIMENT ANALYTICS
# ---------------------------------------------------------
with tab_pca:
    st.subheader("MLflow Multi-Model Candidate Comparison Matrix")
    comparison_data = ml_engine.get_comparison_matrix()
    if comparison_data:
        df_comp = pd.DataFrame(comparison_data)
        st.dataframe(df_comp, use_container_width=True)
    else:
        st.info("Run training pipeline to populate MLflow experiment comparison matrix.")

    st.divider()
    st.subheader("PCA Mathematical Explained Variance Ratio (Scree Plot)")
    pca_metrics = ml_engine.get_pca_metrics()

    if "explained_variance_ratio" in pca_metrics:
        evr = pca_metrics["explained_variance_ratio"]
        cum_evr = pca_metrics["cumulative_variance_ratio"]
        components = [f"PC{i+1}" for i in range(len(evr))]

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.metric("Total Original Features", pca_metrics.get("total_features", 26))
            st.metric("Retained Components (95% Threshold)", pca_metrics.get("retained_components", 14))
            st.metric("Cumulative Variance Retained", f"{pca_metrics.get('cumulative_variance_explained', 0.95)*100:.2f}%")

        with col_p2:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=components, y=evr, name="Individual EVR"))
            fig.add_trace(go.Scatter(x=components, y=cum_evr, name="Cumulative EVR", yaxis="y2", line=dict(color="red", width=3)))

            fig.update_layout(
                title="PCA Scree Plot & Cumulative Variance Curve",
                xaxis_title="Principal Components",
                yaxis_title="Individual Variance Ratio",
                yaxis2=dict(title="Cumulative Variance Ratio", overlaying="y", side="right", range=[0, 1.05]),
                legend=dict(x=0.6, y=0.2)
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# TAB 4: CRYPTOGRAPHIC AUDIT LEDGER
# ---------------------------------------------------------
with tab_audit:
    st.subheader("Tamper-Evident SHA-256 Cryptographic Hash Chain Inspector")

    col_v1, col_v2 = st.columns([1, 3])
    with col_v1:
        if st.button("🛡️ Verify Ledger Integrity", type="primary"):
            validation = orchestrator.verify_audit_chain()
            if validation["is_valid"]:
                st.success("✅ AUDIT LEDGER INTEGRITY VALIDATED")
                st.write(f"**Total Records**: {validation['total_records']}")
                st.write(f"**Tip Hash**: `{validation.get('tip_hash')[:16]}...`")
            else:
                st.error("🚨 AUDIT LEDGER TAMPERING DETECTED!")
                st.write(validation)

    ledger = orchestrator.get_audit_ledger()
    df_ledger = pd.DataFrame(ledger)
    st.dataframe(df_ledger, use_container_width=True)
