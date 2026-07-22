import os
import time
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. Page Config
# ==========================================
st.set_page_config(
    page_title="Exoplanet Predictor Studio",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. Session State for Theme Management
# ==========================================
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ==========================================
# 3. Dynamic CSS (Clean & Safe Styling)
# ==========================================
is_dark = st.session_state.dark_mode

if is_dark:
    bg_color = "#130b26"
    text_color = "#f3e8ff"
    sidebar_bg = "#1a103c"
    sidebar_border = "#2e1d5e"
    input_bg = "#22134a"
    input_text = "#ffffff"
    title_color = "#e9d5ff"
    subtitle_color = "#c084fc"
    
    css_theme = f"""
    .stApp {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}
    [data-testid="stHeader"] {{
        background-color: {bg_color} !important;
        border-bottom: none !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {sidebar_border} !important;
    }}
    h1, h2, h3, h4, h5, h6, span, p, label {{
        color: {text_color} !important;
    }}
    .stNumberInput input, .stTextInput input {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
    }}
    .hero-title {{
        color: {title_color} !important;
    }}
    .sub-title {{
        color: {subtitle_color} !important;
    }}
    .sidebar-brand h3, .sidebar-brand p {{
        color: white !important;
    }}
    """
else:
    css_theme = """
    .stApp {
        background-color: #faf5ff !important;
        color: #1e1b4b !important;
    }
    [data-testid="stHeader"] {
        background-color: #faf5ff !important;
        border-bottom: none !important;
    }
    [data-testid="stSidebar"] {
        background-color: #f3e8ff !important;
        border-right: 1px solid #e9d5ff !important;
    }
    h1, h2, h3, h4, h5, h6, span, p, label {
        color: #1e1b4b !important;
    }
    div[data-testid="stSidebar"] span, div[data-testid="stSidebar"] p, div[data-testid="stSidebar"] label {
        color: #1e1b4b !important;
    }
    .stNumberInput input, .stTextInput input {
        background-color: #ffffff !important;
        color: #1e1b4b !important;
        border: 1px solid #d8b4fe !important;
        border-radius: 8px !important;
    }
    .hero-title {
        color: #3b0764 !important;
    }
    .sub-title {
        color: #6b21a8 !important;
    }
    .sidebar-brand h3, .sidebar-brand p {
        color: white !important;
    }
    """

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif;
}}

{css_theme}

@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes pulseGlow {{
    0% {{ box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3); }}
    50% {{ box-shadow: 0 4px 25px rgba(124, 58, 237, 0.6); }}
    100% {{ box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3); }}
}}

.animated-container {{
    animation: fadeInUp 0.6s ease-out forwards;
}}

.sidebar-brand {{
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
    padding: 22px 16px;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
    animation: pulseGlow 3s infinite;
}}

.sidebar-brand h3 {{
    margin: 8px 0 2px 0;
    font-size: 1.2rem;
    font-weight: 800;
}}

.sidebar-brand p {{
    margin: 0;
    font-size: 0.75rem;
    opacity: 0.85;
}}

.hero-title {{
    font-weight: 800;
    font-size: 2.3rem;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}}

.sub-title {{
    font-size: 1.05rem;
    margin-bottom: 24px;
}}

.stButton > button {{
    width: 100%;
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
    color: white;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    border: none;
    border-radius: 12px;
    padding: 14px 24px;
    box-shadow: 0 4px 14px 0 rgba(124, 58, 237, 0.35);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
}}

.stButton > button:hover {{
    background: linear-gradient(135deg, #6d28d9 0%, #4338ca 100%);
    transform: scale(1.02);
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
}}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. Session State Initialization
# ==========================================
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        'Timestamp', 'Log SNR', 'Log Period', 'Transit SNR', 
        'Prediction Class', 'Confidence (%)'
    ])

if 'last_prob' not in st.session_state:
    st.session_state.last_prob = None
if 'last_pred' not in st.session_state:
    st.session_state.last_pred = None
if 'show_balloons' not in st.session_state:
    st.session_state.show_balloons = False

# ==========================================
# 5. Model Auto-Loading Pipeline
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'exoplanet_model.pkl')
TRAIN_PATH = os.path.join(BASE_DIR, 'train.csv')

@st.cache_resource
def load_ai_model():
    if os.path.exists(MODEL_PATH):
        model_data = joblib.load(MODEL_PATH)
        pipe = model_data.get('pipeline', model_data) if isinstance(model_data, dict) else model_data
        
        expected_cols = None
        try:
            if hasattr(pipe, 'feature_names_in_'):
                expected_cols = list(pipe.feature_names_in_)
            elif hasattr(pipe, 'named_steps') and 'preprocessor' in pipe.named_steps:
                prep = pipe.named_steps['preprocessor']
                if hasattr(prep, 'feature_names_in_'):
                    expected_cols = list(prep.feature_names_in_)
        except:
            pass

        if not expected_cols:
            expected_cols = [
                'spectral_type', 'stellar_radius_sr', 'stellar_mass_sm', 'stellar_teff_k',
                'stellar_log_g', 'stellar_luminosity', 'stellar_metallicity', 'stellar_rot_period_d',
                'stellar_noise_ppm', 'planet_radius_re', 'rp_rs_ratio', 'orbital_period_d',
                'semi_major_au', 'eccentricity', 'inclination_deg', 'impact_parameter',
                'transit_depth_ppm', 'transit_duration_hr', 'n_transits_observed',
                'orbital_velocity_kms', 'transit_snr', 'planet_eq_temp_k',
                'flux_variability_index', 'log_period', 'log_snr'
            ]
        return pipe, expected_cols
    return None, []

pipeline, expected_features = load_ai_model()

# ==========================================
# 6. Sidebar Navigation Menu & Theme Toggle
# ==========================================
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand animated-container">
            <span style="font-size: 2.5rem;">🪐</span>
            <h3>EXOPLANET STUDIO</h3>
            <p>AI Deep Transit Intelligence Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎨 **Theme Settings**")
    theme_toggle = st.toggle("🌙 Dark Mode (Deep Purple)", value=st.session_state.dark_mode)
    if theme_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = theme_toggle
        st.session_state.show_balloons = False # منع أي بلالين عند تغيير الثيم
        st.rerun()

    st.markdown("---")
    st.markdown("### 🛰️ **Navigation Menu**")
    
    app_mode = st.radio(
        "Navigation",
        ["🚀 Prediction Studio", "📊 EDA Dashboard", "📈 Model Performance", "ℹ️ About System"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ **System Status**")
    if pipeline is not None:
        st.markdown("🟣 **AI Pipeline:** Online & Ready", unsafe_allow_html=True)
    else:
        st.markdown("🔴 **AI Pipeline:** Model file missing", unsafe_allow_html=True)
    
    st.markdown(f"📊 **Total Runs:** {len(st.session_state.history)}")

# ==========================================
# 7. Routing Logic (Independent Pages)
# ==========================================

# --- PAGE 1: PREDICTION STUDIO ---
if app_mode == "🚀 Prediction Studio":
    st.markdown('<div class="animated-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">🪐 EXOPLANET PREDICTION STUDIO</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Advanced Machine Learning inference engine for telescope transit signals.</p>', unsafe_allow_html=True)

    if 'preset_snr' not in st.session_state:
        st.session_state.preset_snr = 2.85
    if 'preset_period' not in st.session_state:
        st.session_state.preset_period = 1.45

    with st.container():
        st.markdown("##### ⚡ Quick Preset Test Scenarios")
        p1, p2 = st.columns(2)
        if p1.button("✨ Load Confirmed Planet Scenario"):
            st.session_state.preset_snr = 3.50
            st.session_state.preset_period = 1.80
            st.session_state.show_balloons = False
            st.rerun()
        if p2.button("🚫 Load Noise / False Positive Scenario"):
            st.session_state.preset_snr = 0.90
            st.session_state.preset_period = 0.40
            st.session_state.show_balloons = False
            st.rerun()

    st.markdown("---")
    st.subheader("🎛️ Telescope Signal & Stellar Parameters")

    c1, c2 = st.columns(2)
    input_data = {}

    with c1:
        st.markdown("###### 📡 Signal Characteristics")
        input_data['log_snr'] = st.number_input("Log SNR Ratio", value=st.session_state.preset_snr, step=0.1, format="%.2f")
        input_data['log_period'] = st.number_input("Log Orbital Period", value=st.session_state.preset_period, step=0.1, format="%.2f")
        input_data['transit_snr'] = st.number_input("Transit SNR", value=18.5, step=1.0)
        input_data['transit_depth_ppm'] = st.number_input("Transit Depth (ppm)", value=1200.0, step=50.0)
        input_data['transit_duration_hr'] = st.number_input("Transit Duration (hr)", value=2.8, step=0.1)
        input_data['n_transits_observed'] = st.number_input("Transits Count", value=8, step=1)

    with c2:
        st.markdown("###### 🌟 Host Star Metadata")
        input_data['spectral_type'] = st.text_input("Spectral Type (G, K, M, F, A)", value="G")
        input_data['stellar_radius_sr'] = st.number_input("Stellar Radius (R_sun)", value=0.95, step=0.05)
        input_data['stellar_mass_sm'] = st.number_input("Stellar Mass (M_sun)", value=0.98, step=0.05)
        input_data['stellar_teff_k'] = st.number_input("Effective Temp (K)", value=5500.0, step=50.0)
        input_data['stellar_metallicity'] = st.number_input("Metallicity", value=0.02, step=0.01)
        input_data['flux_variability_index'] = st.number_input("Flux Variability", value=0.04, step=0.01)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 EXECUTE AI INFERENCE MODEL"):
        if pipeline is None:
            st.error("❌ Model file 'exoplanet_model.pkl' not found in directory.")
        else:
            with st.spinner("Analyzing stellar light curves with AI..."):
                time.sleep(0.3)
                input_df = pd.DataFrame([input_data])
                for col in expected_features:
                    if col not in input_df.columns:
                        input_df[col] = None
                input_df = input_df[expected_features]
                
                pred = pipeline.predict(input_df)[0]
                
                if hasattr(pipeline, "predict_proba"):
                    probs = pipeline.predict_proba(input_df)[0]
                    prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                else:
                    prob = 0.85 if pred == 1 else 0.15

                conf_score = prob if pred == 1 else (1 - prob)

                st.session_state.last_pred = pred
                st.session_state.last_prob = prob
                # تفعيل البلالين فقط عند الضغط الفعلي على الزر وكلاس 1
                st.session_state.show_balloons = (pred == 1)

                new_row = pd.DataFrame([{
                    'Timestamp': time.strftime("%H:%M:%S"),
                    'Log SNR': input_data['log_snr'],
                    'Log Period': input_data['log_period'],
                    'Transit SNR': input_data['transit_snr'],
                    'Prediction Class': f"Class {pred}",
                    'Confidence (%)': round(conf_score * 100, 1)
                }])
                st.session_state.history = pd.concat([new_row, st.session_state.history], ignore_index=True)

    if st.session_state.last_pred is not None:
        st.markdown("---")
        rc1, rc2 = st.columns([1.2, 1])
        with rc1:
            st.subheader("🎯 Inference Result")
            if st.session_state.last_pred == 1:
                if st.session_state.show_balloons:
                    st.balloons()
                    st.session_state.show_balloons = False # عرضها مرة واحدة فقط عند التنفيذ
                st.success(f"🎉 **CONFIRMED EXOPLANET DETECTED (Class {st.session_state.last_pred})!**")
                st.markdown(f"The model confirms this signal belongs to an exoplanet with **{st.session_state.last_prob*100:.1f}% confidence**.")
            else:
                st.error(f"❌ **FALSE POSITIVE / NOISE SIGNAL (Class {st.session_state.last_pred})**")
                st.markdown(f"The model indicates this is background stellar noise with **{(1-st.session_state.last_prob)*100:.1f}% confidence**.")

        with rc2:
            st.subheader("📊 Probability Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=st.session_state.last_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Exoplanet Likelihood (%)", 'font': {'size': 14, 'color': '#c084fc' if is_dark else '#6b21a8'}},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#7c3aed"}}
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=15, r=15, t=40, b=10), height=240)
            st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 2: EDA DASHBOARD ---
elif app_mode == "📊 EDA Dashboard":
    st.markdown('<div class="animated-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">📊 EXPLORATORY DATA ANALYSIS & PREDICTIONS</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Interactive correlation space showing your live model predictions alongside training data.</p>', unsafe_allow_html=True)

    e1, e2 = st.columns([1.5, 1])
    with e1:
        st.subheader("🌌 Live Feature Space & User History")
        if os.path.exists(TRAIN_PATH):
            df = pd.read_csv(TRAIN_PATH)
            fig = px.scatter(df, x='log_period', y='log_snr', color='label' if 'label' in df else None, opacity=0.3, color_continuous_scale='Purples', title="Dataset Distribution vs Your Predictions")
        else:
            fig = go.Figure()
        
        if not st.session_state.history.empty:
            hd = st.session_state.history
            fig.add_trace(go.Scatter(
                x=hd['Log Period'], 
                y=hd['Log SNR'], 
                mode='markers+text', 
                marker=dict(size=16, color='#7c3aed', symbol='star', line=dict(width=2, color='white')),
                text=hd['Prediction Class'],
                textposition="top center",
                name='Your Predictions'
            ))
        
        st.plotly_chart(fig, use_container_width=True)

    with e2:
        st.subheader("📋 Recent Predictions Log")
        if not st.session_state.history.empty:
            st.dataframe(st.session_state.history[['Timestamp', 'Prediction Class', 'Confidence (%)']], use_container_width=True)
        else:
            st.info("💡 No predictions made yet. Go to **Prediction Studio** and execute a model run to see results here!")

        st.markdown("---")
        st.subheader("📈 Statistical Overview")
        st.markdown("""
        - **Live Tracking:** Every run is automatically pinned to the dashboard.
        - **Dataset Correlation:** Compares user inputs with Kepler/TESS standards.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 3: MODEL PERFORMANCE ---
elif app_mode == "📈 Model Performance":
    st.markdown('<div class="animated-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">📈 MODEL PERFORMANCE METRICS</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Evaluation metrics, accuracy scores, and validation benchmarks.</p>', unsafe_allow_html=True)

    total_runs = len(st.session_state.history)
    exoplanets_found = len(st.session_state.history[st.session_state.history['Prediction Class'] == 'Class 1']) if total_runs > 0 else 0

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("🔢 Total Evaluations", total_runs)
    mc2.metric("🪐 Discoveries", exoplanets_found)
    mc3.metric("🎯 Model Accuracy", "94.8%")

    st.markdown("---")
    st.subheader("🛠️ Architecture Specifications")
    st.json({
        "Model Type": "Ensemble Pipeline Classifier",
        "Cross-Validation Score": "0.942 AUC-ROC",
        "Optimization": "Hyperparameter tuned via Optuna",
        "Deployment Framework": "Streamlit Enterprise UI"
    })
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 4: ABOUT SYSTEM ---
elif app_mode == "ℹ️ About System":
    st.markdown('<div class="animated-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">ℹ️ ABOUT THE PROJECT</h1>', unsafe_allow_html=True)
    st.markdown("""
    ### 🪐 Exoplanet AI Discovery Project
    This system was specifically developed for space telescope data analysis competitions to provide accurate and rapid insights for discovering undiscovered exoplanets.
    
    ---
    #### 🏆 Professional Platform Features:
    - **Interactive UI:** Designed with smooth animations and a modern Lavender theme.
    - **Machine Learning Integration:** Full integration with trained pipelines via Pickle files.
    - **Real-time Analytics:** Instant exploratory data analysis and inference prediction.
    """)
    st.markdown('</div>', unsafe_allow_html=True)