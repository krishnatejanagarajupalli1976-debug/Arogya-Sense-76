import streamlit as st

from healthcare_model import DISCLAIMER, get_model, model_metadata

st.set_page_config(page_title="ArogyaSense", page_icon="🏥", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f7f4ed; color: #173042; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: #173042; }
.hero { padding: 2rem 0 1rem; border-bottom: 1px solid #dedbd2; }
.eyebrow { color: #087f80; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; font-size: .75rem; }
.result { background: white; border-left: 6px solid #ef705e; padding: 1.25rem; border-radius: 8px; }
.disclaimer { color: #7a6d61; font-size: .85rem; }
</style>
""", unsafe_allow_html=True)

model = get_model()
st.markdown('<div class="hero"><div class="eyebrow">🏥 Nagpur hospital network · clinical decision support</div><h1>ArogyaSense</h1><p>Understand symptom patterns before your clinical consultation.</p></div>', unsafe_allow_html=True)
left, right = st.columns([1.2, 0.8], gap="large")
with left:
    st.subheader("What are you experiencing?")
    selected = st.multiselect("Select all symptoms that apply", model.features, format_func=lambda item: item.replace("_", " ").title())
    if st.button("Analyze symptoms", type="primary", use_container_width=True):
        if not selected:
            st.warning("Select at least one symptom to continue.")
        else:
            st.session_state["prediction"] = model.predict(selected)
    st.caption("Predictions are based on a public symptom-disease training pack and should be reviewed by a clinician.")
with right:
    st.subheader("Model transparency")
    meta = model_metadata()
    st.metric("Diseases covered", meta["disease_count"])
    st.metric("Validation accuracy", f'{meta["validation_accuracy"]:.0%}')
    st.metric("Symptoms available", meta["symptom_count"])

prediction = st.session_state.get("prediction")
if prediction:
    st.divider()
    st.subheader("Screening result")
    st.markdown(f'<div class="result"><div class="eyebrow">Most likely pattern</div><h2>{prediction["disease"]}</h2><strong>{prediction["confidence"]:.0%} model confidence</strong></div>', unsafe_allow_html=True)
    st.write("Other likely patterns")
    st.dataframe(prediction["top_predictions"], hide_index=True, use_container_width=True)
    st.write("Why this result")
    for item in prediction["shap_explanation"]:
        st.progress(min(item["impact"] * 12, 1.0), text=f'{item["symptom"]}: contribution {item["impact"]:.3f}')
    st.markdown(f'<p class="disclaimer">{DISCLAIMER}</p>', unsafe_allow_html=True)
