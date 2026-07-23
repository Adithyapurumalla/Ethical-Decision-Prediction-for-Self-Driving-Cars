"""
Self-Driving Car Ethics AI - Interactive Studio & AI Simulator

A high-tech, creative, and user-friendly web application for simulating,
visualizing, and explaining moral and ethical dilemmas in autonomous vehicles.
"""

import sys
import io
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

# ==============================================================================
# PAGE CONFIGURATION & ADVANCED STUDIO STYLING
# ==============================================================================

st.set_page_config(
    page_title="Ethical Decision Prediction for Self-Driving Cars",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Studio Aesthetic
def inject_custom_css():
    st.markdown("""
    <style>
        /* Global Background & Typography */
        .stApp {
            background: #090d16;
            color: #f1f5f9;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }

        /* Top Hero Header Banner */
        .hero-banner {
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 50%, #030712 100%);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 25px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(99, 102, 241, 0.1);
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 10px 0;
        }
        .hero-subtitle {
            color: #94a3b8;
            font-size: 1.1rem;
            margin: 0;
        }

        /* Preset Card Buttons */
        .preset-btn {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .preset-btn:hover {
            border-color: #38bdf8;
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(56, 189, 248, 0.2);
        }

        /* Decision Action Badges */
        .badge-decision {
            font-size: 1.6rem;
            font-weight: 800;
            padding: 12px 24px;
            border-radius: 50px;
            display: inline-block;
            letter-spacing: 1px;
            text-transform: uppercase;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
        }
        .badge-brake_hard {
            background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
            color: white;
            border: 1px solid #f87171;
        }
        .badge-swerve_left {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
            border: 1px solid #fbbf24;
        }
        .badge-swerve_right {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: white;
            border: 1px solid #60a5fa;
        }
        .badge-maintain_course {
            background: linear-gradient(135deg, #10b981 0%, #047857 100%);
            color: white;
            border: 1px solid #34d399;
        }

        /* Glassmorphic Container Cards */
        .glass-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        }
        .glass-card h3 {
            color: #38bdf8;
            margin-top: 0;
            font-size: 1.25rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Explanation Callout */
        .explanation-callout {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(129, 140, 248, 0.1) 100%);
            border-left: 4px solid #38bdf8;
            border-radius: 8px;
            padding: 18px;
            font-size: 1.05rem;
            line-height: 1.6;
            color: #e2e8f0;
        }

        /* Sidebar Customization */
        section[data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right: 1px solid #1e293b;
        }

        /* Tab Bar Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: #1e293b;
            padding: 8px;
            border-radius: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            border-radius: 8px;
            color: #94a3b8;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #38bdf8 !important;
            color: #0f172a !important;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# Initialize session state for preset inputs
if "sim_params" not in st.session_state:
    st.session_state.sim_params = {
        "speed_mph": 45,
        "pedestrian_count": 3,
        "passenger_count": 1,
        "obstacle_type": "pedestrian_group",
        "brake_status": "failed",
        "weather_condition": "clear",
        "road_type": "city_street",
        "time_of_day": "day",
        "pedestrian_jaywalking": 0,
        "ethical_score": 0.85
    }

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ==============================================================================
# CACHED LOADERS
# ==============================================================================

@st.cache_resource
def load_ml_pipeline() -> Tuple[Optional[Any], Optional[Any]]:
    """Loads trained pipeline and preprocessor."""
    model_path = Path("models/best_model.pkl")
    preprocessor_path = Path("models/preprocessor.pkl")
    model, preprocessor = None, None
    if model_path.exists():
        try:
            model = joblib.load(model_path)
        except Exception:
            pass
    if preprocessor_path.exists():
        try:
            preprocessor = joblib.load(preprocessor_path)
        except Exception:
            pass
    return model, preprocessor


@st.cache_data
def load_datasets() -> Dict[str, pd.DataFrame]:
    """Loads CSV datasets from data/ with memory optimization for cloud deployment."""
    data_dir = Path("data")
    datasets = {}
    if data_dir.exists():
        for csv_path in data_dir.glob("*.csv"):
            try:
                if csv_path.name == "moral_machine_responses.csv":
                    datasets[csv_path.name] = pd.read_csv(csv_path, nrows=10000)
                else:
                    datasets[csv_path.name] = pd.read_csv(csv_path)
            except Exception:
                pass
    return datasets


@st.cache_data
def load_metrics_and_report() -> Tuple[Optional[pd.DataFrame], str]:
    """Loads metrics CSV and report text."""
    metrics_path = Path("models/model_metrics.csv")
    report_path = Path("outputs/models/model_report.txt")
    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else None
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    return metrics_df, report_text


model, preprocessor = load_ml_pipeline()
datasets = load_datasets()
metrics_df, report_text = load_metrics_and_report()


# ==============================================================================
# PRESET SCENARIOS APPLY FUNCTION
# ==============================================================================

def apply_preset(preset_name: str):
    """Sets session state simulation parameters to a preset dilemma archetype."""
    if preset_name == "school_zone":
        st.session_state.sim_params = {
            "speed_mph": 30, "pedestrian_count": 5, "passenger_count": 1,
            "obstacle_type": "pedestrian_group", "brake_status": "failed",
            "weather_condition": "clear", "road_type": "school_zone",
            "time_of_day": "day", "pedestrian_jaywalking": 0, "ethical_score": 0.95
        }
    elif preset_name == "barricade":
        st.session_state.sim_params = {
            "speed_mph": 60, "pedestrian_count": 4, "passenger_count": 2,
            "obstacle_type": "barricade", "brake_status": "failed",
            "weather_condition": "rain", "road_type": "highway",
            "time_of_day": "night", "pedestrian_jaywalking": 0, "ethical_score": 0.70
        }
    elif preset_name == "animal":
        st.session_state.sim_params = {
            "speed_mph": 45, "pedestrian_count": 1, "passenger_count": 3,
            "obstacle_type": "animal", "brake_status": "functional",
            "weather_condition": "clear", "road_type": "city_street",
            "time_of_day": "dusk", "pedestrian_jaywalking": 0, "ethical_score": 0.90
        }
    elif preset_name == "jaywalker":
        st.session_state.sim_params = {
            "speed_mph": 35, "pedestrian_count": 1, "passenger_count": 2,
            "obstacle_type": "jaywalker", "brake_status": "functional",
            "weather_condition": "fog", "road_type": "city_street",
            "time_of_day": "night", "pedestrian_jaywalking": 1, "ethical_score": 0.65
        }


# ==============================================================================
# VISUAL SCENARIO SVG GRAPHIC RENDERER
# ==============================================================================

def render_scenario_svg(params: Dict[str, Any], decision: str) -> str:
    """
    Renders an interactive SVG diagram representing the autonomous vehicle,
    road environment, obstacles, and predicted maneuver trajectory.
    """
    brakes = params.get("brake_status", "functional")
    p_count = params.get("pedestrian_count", 1)
    pass_count = params.get("passenger_count", 1)
    obstacle = params.get("obstacle_type", "pedestrian_group")
    speed = params.get("speed_mph", 40)

    # Determine trajectory visual path
    path_d = "M 200 350 L 200 180"  # Straight
    stroke_color = "#10b981"
    action_text = "MAINTAIN COURSE"

    if decision == "brake_hard":
        path_d = "M 200 350 L 200 250"
        stroke_color = "#ef4444"
        action_text = "BRAKE HARD 🛑"
    elif decision == "swerve_left":
        path_d = "M 200 350 Q 200 240 100 150"
        stroke_color = "#f59e0b"
        action_text = "SWERVE LEFT ↙️"
    elif decision == "swerve_right":
        path_d = "M 200 350 Q 200 240 300 150"
        stroke_color = "#3b82f6"
        action_text = "SWERVE RIGHT ↘️"

    obstacle_icon = "🚶‍♂️🚶‍♀️" if obstacle in ["pedestrian_group", "jaywalker"] else "🧱" if obstacle == "barricade" else "🦌" if obstacle == "animal" else "🚘"

    svg_code = f"""
    <svg viewBox="0 0 400 400" style="width: 100%; max-height: 320px; background: #0f172a; border-radius: 12px; border: 1px solid #1e293b;">
        <!-- Road Background -->
        <rect x="60" y="0" width="280" height="400" fill="#1e293b" />
        
        <!-- Lane Markings -->
        <line x1="200" y1="0" x2="200" y2="400" stroke="#475569" stroke-width="4" stroke-dasharray="15,15" />
        <line x1="60" y1="0" x2="60" y2="400" stroke="#f59e0b" stroke-width="4" />
        <line x1="340" y1="0" x2="340" y2="400" stroke="#f59e0b" stroke-width="4" />

        <!-- Trajectory Arrow -->
        <path d="{path_d}" fill="none" stroke="{stroke_color}" stroke-width="6" stroke-linecap="round" marker-end="url(#arrow)" />

        <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="{stroke_color}" />
            </marker>
        </defs>

        <!-- Obstacle Zone -->
        <g transform="translate(160, 80)">
            <rect x="-10" y="-10" width="100" height="50" rx="8" fill="rgba(239, 68, 68, 0.2)" stroke="#ef4444" stroke-width="2"/>
            <text x="40" y="22" font-size="22" text-anchor="middle">{obstacle_icon}</text>
            <text x="40" y="55" font-size="11" fill="#f87171" font-weight="bold" text-anchor="middle">{obstacle.upper().replace('_', ' ')} (x{p_count})</text>
        </g>

        <!-- Autonomous Vehicle -->
        <g transform="translate(165, 290)">
            <rect x="0" y="0" width="70" height="90" rx="12" fill="#0284c7" stroke="#38bdf8" stroke-width="3"/>
            <!-- Windshield -->
            <rect x="8" y="15" width="54" height="25" rx="4" fill="#0f172a" />
            <!-- Occupant Indicator -->
            <text x="35" y="32" font-size="14" fill="#38bdf8" text-anchor="middle">👤 x{pass_count}</text>
            <!-- Headlights -->
            <circle cx="12" cy="5" r="4" fill="#fef08a" />
            <circle cx="58" cy="5" r="4" fill="#fef08a" />
            <text x="35" y="70" font-size="11" fill="white" font-weight="bold" text-anchor="middle">{speed} MPH</text>
        </g>

        <!-- Action Overlay -->
        <rect x="20" y="20" width="160" height="35" rx="6" fill="rgba(15, 23, 42, 0.85)" stroke="{stroke_color}" stroke-width="1.5"/>
        <text x="100" y="42" font-size="12" fill="{stroke_color}" font-weight="bold" text-anchor="middle">{action_text}</text>
    </svg>
    """
    return svg_code


# ==============================================================================
# NARRATIVE EXPLAINER
# ==============================================================================

def generate_narrative_explanation(params: Dict[str, Any], decision: str, confidence: float) -> str:
    """Generates human-friendly narrative explanation for AI choice."""
    brakes = params.get("brake_status", "functional")
    p_count = params.get("pedestrian_count", 1)
    pass_count = params.get("passenger_count", 1)
    obstacle = params.get("obstacle_type", "pedestrian_group")
    jaywalking = params.get("pedestrian_jaywalking", 0)

    reasons = []
    if brakes == "failed":
        reasons.append("the mechanical brakes have failed, preventing normal stopping")
    else:
        reasons.append("brakes are operational, allowing rapid deceleration")

    if p_count > pass_count:
        reasons.append(f"pedestrians ({p_count}) outnumber vehicle occupants ({pass_count})")
    elif pass_count > p_count:
        reasons.append(f"vehicle occupants ({pass_count}) outnumber pedestrians ({p_count})")

    if obstacle == "pedestrian_group":
        reasons.append("a high-priority group of human pedestrians is in the direct line of travel")
    elif obstacle == "school_zone":
        reasons.append("the vehicle is within a school safety zone with high vulnerability risk")

    if jaywalking == 1:
        reasons.append("the pedestrians crossed against the legal traffic signal")

    explanation = (
        f"<strong>AI Decision Narrative:</strong> With <strong>{confidence:.1f}% model confidence</strong>, "
        f"the autonomous vehicle chose to <strong>'{decision.upper().replace('_', ' ')}'</strong> because " + "; ".join(reasons) + ". "
        f"This decision prioritizes minimizing total casualties based on global survey preferences."
    )
    return explanation


# ==============================================================================
# HERO HEADER & TOP NAVIGATION
# ==============================================================================

st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">🚗 Ethical Decision Prediction for Self-Driving Cars</h1>
    <p class="hero-subtitle">Interactive Moral Machine Dilemma Simulator & Decision Explainer</p>
</div>
""", unsafe_allow_html=True)

# Main Workspace Tabs
tab_studio, tab_batch, tab_analytics, tab_performance, tab_history = st.tabs([
    "🕹️ Interactive Scenario Studio",
    "📁 Batch Scenario Upload",
    "📊 Visual Analytics & Insights",
    "⚡ Model Benchmark & Intelligence",
    "📜 Session History Log"
])


# ==============================================================================
# TAB 1: INTERACTIVE SCENARIO STUDIO
# ==============================================================================

with tab_studio:
    st.markdown("### ⚡ Quick Scenario Archetype Presets")
    st.markdown("Click any preset below to instantly load a classic moral dilemma into the simulation engine:")

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        if st.button("🚸 School Zone Dilemma"):
            apply_preset("school_zone")
            st.rerun()
    with col_p2:
        if st.button("🧱 High-Speed Wall Impact"):
            apply_preset("barricade")
            st.rerun()
    with col_p3:
        if st.button("🦌 Animal Highway Crossing"):
            apply_preset("animal")
            st.rerun()
    with col_p4:
        if st.button("🚦 Night Jaywalker Scenario"):
            apply_preset("jaywalker")
            st.rerun()

    st.markdown("---")

    # Main Interactive Control & Simulation Layout
    ctrl_col, sim_col = st.columns([1, 1])

    with ctrl_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Live Scenario Control Panel")

        p = st.session_state.sim_params

        # Real-time sliders & selects (No submit button needed!)
        p["speed_mph"] = st.slider("Vehicle Speed (MPH)", 10, 90, p["speed_mph"], step=5, key="sp_slider")
        
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            p["brake_status"] = st.radio("Brake System Status", ["functional", "failed"], index=0 if p["brake_status"] == "functional" else 1, key="brk_radio")
            p["pedestrian_count"] = st.number_input("Pedestrian Count", 1, 10, p["pedestrian_count"], key="ped_num")
            p["passenger_count"] = st.number_input("Vehicle Occupant Count", 1, 6, p["passenger_count"], key="pass_num")
        with c_sub2:
            p["obstacle_type"] = st.selectbox("Obstacle / Target Type", ["pedestrian_group", "barricade", "animal", "vehicle", "jaywalker"], index=0, key="obs_select")
            p["road_type"] = st.selectbox("Road Setting", ["city_street", "highway", "school_zone", "intersection"], index=0, key="road_select")
            p["weather_condition"] = st.selectbox("Weather Condition", ["clear", "rain", "fog", "snow"], index=0, key="wtr_select")

        c_sub3, c_sub4 = st.columns(2)
        with c_sub3:
            p["time_of_day"] = st.selectbox("Time of Day", ["day", "night", "dusk"], index=0, key="time_select")
        with c_sub4:
            p["pedestrian_jaywalking"] = st.selectbox("Crossing Signal", [0, 1], format_func=lambda x: "Legal Signal (0)" if x == 0 else "Jaywalking (1)", key="jay_select")

        p["ethical_score"] = st.slider("Ethical Priority Index", 0.0, 1.0, float(p["ethical_score"]), step=0.05, key="eth_slider")

        st.markdown('</div>', unsafe_allow_html=True)

    with sim_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Live Simulation & AI Decision")

        if model is None:
            st.error("⚠️ Trained model not found! Please run `train_model.py` to activate ML prediction.")
        else:
            # Predict instantly based on parameters
            input_df = pd.DataFrame([p])
            proba = model.predict_proba(input_df)[0]
            pred_idx = int(np.argmax(proba))

            classifier = model.named_steps["classifier"]
            classes = list(getattr(classifier, "classes_", [0, 1, 2, 3]))
            label_map = {0: "brake_hard", 1: "maintain_course", 2: "swerve_left", 3: "swerve_right"}

            if isinstance(classes[0], (int, np.integer)):
                class_labels = [label_map.get(c, str(c)) for c in classes]
            else:
                class_labels = [str(c) for c in classes]

            pred_decision = class_labels[pred_idx]
            confidence = float(proba[pred_idx] * 100)

            # Record history entry
            history_entry = {**p, "Predicted_Decision": pred_decision, "Confidence_%": round(confidence, 2)}
            if not st.session_state.prediction_history or st.session_state.prediction_history[-1] != history_entry:
                st.session_state.prediction_history.append(history_entry)

            # Display Decision Action Badge
            st.markdown(f'<div style="text-align: center; margin: 15px 0;"><span class="badge-decision badge-{pred_decision}">{pred_decision.upper().replace("_", " ")}</span></div>', unsafe_allow_html=True)
            st.markdown(f"**Model Confidence:** `{confidence:.1f}%`")
            st.progress(confidence / 100)

            # Visual SVG Road Diagram
            svg_html = render_scenario_svg(p, pred_decision)
            st.components.v1.html(svg_html, height=330)

            st.markdown('</div>', unsafe_allow_html=True)

    # Narrative Explainability & Probability Breakdown Row
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 💡 AI Explainability & Decision Rationale")

    exp_col1, exp_col2 = st.columns([1.2, 1])
    with exp_col1:
        narrative = generate_narrative_explanation(p, pred_decision, confidence)
        st.markdown(f'<div class="explanation-callout">{narrative}</div>', unsafe_allow_html=True)

    with exp_col2:
        st.markdown("#### Probability Breakdown")
        proba_df = pd.DataFrame({
            "Decision": [c.replace("_", " ").title() for c in class_labels],
            "Probability (%)": [round(pr * 100, 2) for pr in proba]
        }).sort_values("Probability (%)", ascending=True)

        fig, ax = plt.subplots(figsize=(6, 2.5))
        sns.barplot(data=proba_df, x="Probability (%)", y="Decision", hue="Decision", legend=False, palette="Blues_d", ax=ax)
        ax.set_xlim(0, 100)
        for patch in ax.patches:
            ax.annotate(f"{patch.get_width():.1f}%", (patch.get_width() + 2, patch.get_y() + patch.get_height() / 2),
                        ha="left", va="center", fontsize=9, color="white", fontweight="bold")
        ax.set_facecolor("#1e293b")
        fig.patch.set_facecolor("#1e293b")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 2: BATCH SCENARIO UPLOAD
# ==============================================================================

with tab_batch:
    st.markdown("### 📁 Batch Multi-Scenario Inference")
    st.markdown("Upload a CSV containing multiple scenario telemetry rows to process bulk predictions.")

    if model is None:
        st.error("⚠️ ML Pipeline binary not loaded.")
    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="batch_uploader")

        # Template Generator
        template_df = pd.DataFrame([{
            "speed_mph": 45, "pedestrian_count": 3, "passenger_count": 1,
            "obstacle_type": "pedestrian_group", "pedestrian_jaywalking": 0,
            "weather_condition": "clear", "brake_status": "failed",
            "road_type": "city_street", "time_of_day": "day", "ethical_score": 0.85
        }, {
            "speed_mph": 25, "pedestrian_count": 1, "passenger_count": 4,
            "obstacle_type": "barricade", "pedestrian_jaywalking": 1,
            "weather_condition": "rain", "brake_status": "functional",
            "road_type": "school_zone", "time_of_day": "night", "ethical_score": 0.90
        }])
        buf = io.StringIO()
        template_df.to_csv(buf, index=False)
        st.download_button("📥 Download Sample Scenario Template", buf.getvalue(), "sample_template.csv", "text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file:
            try:
                b_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {b_df.shape[0]} scenarios.")
                st.dataframe(b_df.head(10), use_container_width=True)

                if st.button("⚡ Run Bulk Predictions", key="run_bulk"):
                    b_preds = model.predict(b_df)
                    b_probas = model.predict_proba(b_df)
                    max_p = np.max(b_probas, axis=1) * 100

                    classifier = model.named_steps["classifier"]
                    classes = list(getattr(classifier, "classes_", [0, 1, 2, 3]))
                    label_map = {0: "brake_hard", 1: "maintain_course", 2: "swerve_left", 3: "swerve_right"}

                    if isinstance(classes[0], (int, np.integer)):
                        b_labels = [label_map.get(p, str(p)) for p in b_preds]
                    else:
                        b_labels = [str(p) for p in b_preds]

                    out_df = b_df.copy()
                    out_df["Predicted_Decision"] = b_labels
                    out_df["Confidence_%"] = np.round(max_p, 2)

                    st.markdown("#### Batch Results Preview")
                    st.dataframe(out_df, use_container_width=True)

                    res_buf = io.StringIO()
                    out_df.to_csv(res_buf, index=False)
                    st.download_button("💾 Export Results CSV", res_buf.getvalue(), "batch_predictions_results.csv", "text/csv")
            except Exception as e:
                st.error(f"Error processing CSV: {e}")


# ==============================================================================
# TAB 3: VISUAL ANALYTICS & INSIGHTS
# ==============================================================================

with tab_analytics:
    st.markdown("### 📊 Global Moral Machine Survey Insights & Analytics")

    if datasets:
        d_name = st.selectbox("Inspect Dataset Source", list(datasets.keys()))
        if d_name in datasets:
            st.dataframe(datasets[d_name].head(15), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🖼️ EDA Chart Gallery (`outputs/eda/`)")

    eda_dir = Path("outputs/eda")
    if eda_dir.exists():
        png_list = sorted([p.name for p in eda_dir.glob("*.png")])
        if png_list:
            sel_png = st.selectbox("Select EDA Visualization", png_list, key="eda_png_select")
            st.image(str(eda_dir / sel_png), caption=sel_png, use_container_width=True)


# ==============================================================================
# TAB 4: MODEL BENCHMARK & INTELLIGENCE
# ==============================================================================

with tab_performance:
    st.markdown("### ⚡ Trained Model Performance & Evaluation Artifacts")

    if metrics_df is not None:
        st.dataframe(metrics_df, use_container_width=True)

    m_out_dir = Path("outputs/models")
    if m_out_dir.exists():
        mc1, mc2 = st.columns(2)
        with mc1:
            if (m_out_dir / "confusion_matrix_best.png").exists():
                st.image(str(m_out_dir / "confusion_matrix_best.png"), caption="Confusion Matrix", use_container_width=True)
            if (m_out_dir / "feature_importance_best.png").exists():
                st.image(str(m_out_dir / "feature_importance_best.png"), caption="Feature Importances", use_container_width=True)
        with mc2:
            if (m_out_dir / "model_comparison.png").exists():
                st.image(str(m_out_dir / "model_comparison.png"), caption="Model Benchmark Comparison", use_container_width=True)
            if (m_out_dir / "roc_curve_best.png").exists():
                st.image(str(m_out_dir / "roc_curve_best.png"), caption="ROC Curve Analysis", use_container_width=True)

    if report_text:
        with st.expander("📜 View Training Report (model_report.txt)"):
            st.code(report_text, language="text")


# ==============================================================================
# TAB 5: SESSION HISTORY LOG
# ==============================================================================

with tab_history:
    st.markdown("### 📜 Session Prediction History")

    if not st.session_state.prediction_history:
        st.info("No predictions recorded in current session. Adjust controls in 'Interactive Scenario Studio' to generate logs.")
    else:
        h_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(h_df, use_container_width=True)

        h_buf = io.StringIO()
        h_df.to_csv(h_buf, index=False)
        st.download_button("💾 Download Session History CSV", h_buf.getvalue(), "session_prediction_history.csv", "text/csv")
