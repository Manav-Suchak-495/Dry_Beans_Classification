import pickle
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Dry Bean Variety Predictor",
    page_icon="🌱",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #066f24;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURE_COLS = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

# Baseline median defaults and safe slider/input bounds based on feature distributions
FEATURE_BOUNDS = {
    "Area": {"min": 20400.0, "max": 254600.0, "default": 44652.0, "step": 100.0},
    "Perimeter": {"min": 524.0, "max": 1975.0, "default": 794.94, "step": 1.0},
    "MajorAxisLength": {"min": 183.0, "max": 738.0, "default": 296.88, "step": 1.0},
    "MinorAxisLength": {"min": 122.0, "max": 460.0, "default": 192.43, "step": 1.0},
    "AspectRation": {"min": 1.024, "max": 2.43, "default": 1.5511, "step": 0.001},
    "Eccentricity": {"min": 0.218, "max": 0.912, "default": 0.7644, "step": 0.001},
    "ConvexArea": {"min": 20684.0, "max": 263261.0, "default": 45178.0, "step": 100.0},
    "EquivDiameter": {"min": 160.0, "max": 570.0, "default": 238.44, "step": 0.1},
    "Extent": {"min": 0.555, "max": 0.866, "default": 0.7599, "step": 0.001},
    "Solidity": {"min": 0.919, "max": 0.994, "default": 0.9883, "step": 0.0001},
    "roundness": {"min": 0.485, "max": 0.995, "default": 0.8832, "step": 0.001},
    "Compactness": {"min": 0.640, "max": 0.990, "default": 0.8013, "step": 0.001},
    "ShapeFactor1": {"min": 0.0027, "max": 0.0105, "default": 0.0066, "step": 0.0001},
    "ShapeFactor2": {"min": 0.0005, "max": 0.0037, "default": 0.0017, "step": 0.0001},
    "ShapeFactor3": {"min": 0.410, "max": 0.980, "default": 0.6420, "step": 0.001},
    "ShapeFactor4": {"min": 0.947, "max": 0.999, "default": 0.9964, "step": 0.0001},
}

st.title("Dry Bean Variety Predictor")
st.caption("Input geometric features below to scale and classify using your saved pickle artifacts.")

with st.form("prediction_form"):
    input_values = {}
    cols = st.columns(4)

    for idx, feat in enumerate(FEATURE_COLS):
        col = cols[idx % 4]
        cfg = FEATURE_BOUNDS.get(
            feat, {"min": 0.0, "max": 1000000.0, "default": 0.0, "step": 1.0}
        )

        input_values[feat] = col.number_input(
            label=feat,
            min_value=float(cfg["min"] * 0.5),
            max_value=float(cfg["max"] * 1.5),
            value=float(cfg["default"]),
            step=float(cfg["step"]),
            format="%.4f" if cfg["max"] < 2.0 else "%.2f",
        )

    submit_btn = st.form_submit_button("Classify Bean", type="primary")

if submit_btn:
    input_df = pd.DataFrame([input_values])[FEATURE_COLS]
    try:
        with open("Dry_Beans_Model.pkl", "rb") as file:
            loaded_pipeline = pickle.load(file)
            
        # Run prediction
        Transformed_Data = loaded_pipeline["CT"].transform(input_df)
        prediction_code = loaded_pipeline["model"].predict(Transformed_Data)[0]
        prediction = loaded_pipeline["mapping"][prediction_code]
        st.success(f"Predicted Dry Bean Variety: **{prediction}**")
        
        if hasattr(loaded_pipeline, "predict_proba"):
            probabilities = loaded_pipeline.predict_proba(input_df)[0]
            classes = loaded_pipeline.classes_
            
            prob_df = pd.DataFrame({
                "Variety": classes,
                "Probability": probabilities
            }).sort_values(by="Probability", ascending=True)
            
            st.subheader("Class Confidence Scores")
            fig = px.bar(
                prob_df,
                x="Probability",
                y="Variety",
                orientation="h",
                text_auto=True,
                labels={"Probability": "Confidence", "Variety": "Bean Variety"},
                title="Model Confidence Distribution"
            )
            fig.update_layout(xaxis_tickformat='%')
            st.plotly_chart(fig, use_container_width=True)
            
    except FileNotFoundError:
        st.error("Model file `Dry_Beans_Model_Pipeline.pkl` not found in the app directory.")
        st.info("Please place your trained pipeline pickle file in the root folder alongside `app.py`.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.stop()