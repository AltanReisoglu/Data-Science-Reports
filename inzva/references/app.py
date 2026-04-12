"""
Asteroid Hazard Predictor
Applied AI Study Group #10 — Week 1

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Asteroid Hazard Predictor",
    page_icon="☄️",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("asteroid_model.pkl")
    features = joblib.load("asteroid_features.pkl")
    return model, features

@st.cache_data
def load_data():
    return pd.read_csv("asteroids.csv")

model, FEATURES = load_model()
df = load_data()

# ── Header ────────────────────────────────────────────────────
st.title("☄️ Asteroid Hazard Predictor")
st.markdown("**Applied AI Study Group #10 — Week 1 Workshop**")
st.markdown("---")

# ── Layout: two columns ───────────────────────────────────────
col_input, col_result = st.columns([1, 1])

with col_input:
    st.subheader("🔬 Asteroid Parameters")
    st.markdown("Adjust the sliders to describe an asteroid:")

    diameter = st.slider(
        "Estimated Diameter (km)",
        min_value=0.01, max_value=5.0, value=0.5, step=0.01,
        help="Larger asteroids are generally more dangerous"
    )
    velocity = st.slider(
        "Relative Velocity (km/s)",
        min_value=1.0, max_value=40.0, value=15.0, step=0.5,
        help="Speed of asteroid relative to Earth"
    )
    miss_distance = st.slider(
        "Miss Distance (million km)",
        min_value=0.1, max_value=100.0, value=20.0, step=0.1,
        help="Closest approach distance to Earth"
    )
    magnitude = st.slider(
        "Absolute Magnitude",
        min_value=14.0, max_value=26.0, value=20.0, step=0.1,
        help="Lower magnitude = brighter = larger/closer"
    )
    eccentricity = st.slider(
        "Orbital Eccentricity",
        min_value=0.0, max_value=0.99, value=0.4, step=0.01,
        help="0 = circular orbit, close to 1 = highly elliptical"
    )
    inclination = st.slider(
        "Orbital Inclination (degrees)",
        min_value=0.0, max_value=30.0, value=10.0, step=0.5,
        help="Angle between asteroid orbit and Earth's orbital plane"
    )

# ── Prediction ────────────────────────────────────────────────
input_data = pd.DataFrame([{
    'est_diameter_km': diameter,
    'relative_velocity_km_s': velocity,
    'miss_distance_mKm': miss_distance,
    'absolute_magnitude': magnitude,
    'eccentricity': eccentricity,
    'inclination_deg': inclination
}])

proba = model.predict_proba(input_data)[0][1]
prediction = model.predict(input_data)[0]

with col_result:
    st.subheader("🎯 Prediction")

    # Big result
    if prediction == 1:
        st.error(f"### ⚠️ HAZARDOUS")
        st.markdown(f"**Hazard Probability: {proba:.1%}**")
    else:
        st.success(f"### ✅ NOT HAZARDOUS")
        st.markdown(f"**Hazard Probability: {proba:.1%}**")

    # ── Interactive Probability Gauge (Plotly) ────────────────
    fig_gauge = go.Figure()

    bar_color = "tomato" if proba > 0.5 else "steelblue"

    fig_gauge.add_trace(go.Bar(
        x=[proba],
        y=["Risk"],
        orientation='h',
        marker_color=bar_color,
        name="Hazard Probability",
        hovertemplate="Hazard Probability: %{x:.1%}<extra></extra>",
        width=0.4
    ))
    fig_gauge.add_trace(go.Bar(
        x=[1 - proba],
        y=["Risk"],
        orientation='h',
        marker_color="#eee",
        name="Safe Probability",
        hovertemplate="Safe Probability: %{x:.1%}<extra></extra>",
        width=0.4
    ))

    fig_gauge.update_layout(
        barmode='stack',
        xaxis=dict(range=[0, 1], title="Hazard Probability"),
        yaxis=dict(showticklabels=False),
        title=dict(text=f"Risk Score: {proba:.1%}", font=dict(size=14)),
        height=150,
        margin=dict(l=10, r=10, t=40, b=30),
        showlegend=False,
        shapes=[
            dict(type="line", x0=0.5, x1=0.5, y0=-0.5, y1=0.5,
                 line=dict(color="gray", dash="dash", width=1))
        ]
    )

    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    # ── Interactive Feature Comparison (Plotly) ───────────────
    st.subheader("📊 How does this compare?")

    feat_labels = {
        'est_diameter_km': ('Diameter (km)', diameter),
        'relative_velocity_km_s': ('Velocity (km/s)', velocity),
        'miss_distance_mKm': ('Miss Distance (mKm)', miss_distance),
        'absolute_magnitude': ('Magnitude', magnitude),
        'eccentricity': ('Eccentricity', eccentricity),
        'inclination_deg': ('Inclination (°)', inclination),
    }

    fig_compare = make_subplots(
        rows=2, cols=3,
        subplot_titles=[feat_labels[f][0] for f in feat_labels],
        horizontal_spacing=0.08,
        vertical_spacing=0.15
    )

    for idx, (feat, (label, user_val)) in enumerate(feat_labels.items()):
        row = idx // 3 + 1
        col = idx % 3 + 1

        # Not hazardous histogram
        not_haz_data = df[df.is_hazardous == 0][feat]
        haz_data = df[df.is_hazardous == 1][feat]

        fig_compare.add_trace(
            go.Histogram(
                x=not_haz_data,
                nbinsx=20,
                opacity=0.5,
                marker_color='steelblue',
                name='Not Haz.',
                showlegend=(idx == 0),
                legendgroup='not_haz',
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Status: Not Hazardous<br>"
                    "Value: %{x}<br>"
                    "Count: %{y}<extra></extra>"
                )
            ),
            row=row, col=col
        )

        # Hazardous histogram
        fig_compare.add_trace(
            go.Histogram(
                x=haz_data,
                nbinsx=20,
                opacity=0.5,
                marker_color='tomato',
                name='Hazardous',
                showlegend=(idx == 0),
                legendgroup='haz',
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Status: Hazardous<br>"
                    "Value: %{x}<br>"
                    "Count: %{y}<extra></extra>"
                )
            ),
            row=row, col=col
        )

        # User input line (highlighted)
        fig_compare.add_vline(
            x=user_val,
            line=dict(color="black", width=2.5, dash="solid"),
            annotation=dict(
                text=f"Your input: {user_val}",
                font=dict(size=9, color="black"),
                bgcolor="rgba(255,255,255,0.8)"
            ),
            row=row, col=col
        )

    fig_compare.update_layout(
        barmode='overlay',
        height=500,
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )

    st.plotly_chart(fig_compare, use_container_width=True)

# ── NEW VISUAL 1: Hazardous vs Not Hazardous Distribution ─────
st.markdown("---")
st.subheader("🧩 Understanding the Data")

col_v1, col_v2 = st.columns([1, 1])

with col_v1:
    st.markdown("#### Hazardous vs Not Hazardous")
    st.markdown("_How many asteroids in our dataset are actually dangerous?_")

    haz_count = df.is_hazardous.sum()
    not_haz_count = len(df) - haz_count

    fig_donut = go.Figure(data=[go.Pie(
        labels=['Not Hazardous', 'Hazardous'],
        values=[not_haz_count, haz_count],
        hole=0.5,
        marker_colors=['steelblue', 'tomato'],
        textinfo='label+percent',
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Count: %{value}<br>"
            "Percentage: %{percent}<extra></extra>"
        )
    )])

    fig_donut.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        annotations=[dict(
            text=f"{haz_count}/{len(df)}",
            x=0.5, y=0.5,
            font_size=18, showarrow=False,
            font_color="tomato"
        )]
    )

    st.plotly_chart(fig_donut, use_container_width=True)

    # Plain-language risk explanation
    hazard_rate = df.is_hazardous.mean()
    st.info(
        f"📌 **Key Insight:** About **{hazard_rate:.0%}** of asteroids in our dataset "
        f"are classified as hazardous. This means roughly **1 in {int(1/hazard_rate)}** "
        f"asteroids pose a potential threat to Earth."
    )

# ── NEW VISUAL 2: Top Features Summary ────────────────────────
with col_v2:
    st.markdown("#### What Makes an Asteroid Dangerous?")
    st.markdown("_The model uses these features to predict hazard level, ranked by importance:_")

    # Get feature importances
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': importances
    }).sort_values('Importance', ascending=True)

    # Plain-language descriptions
    feat_descriptions = {
        'est_diameter_km': '🔵 Estimated Diameter',
        'relative_velocity_km_s': '🟡 Relative Velocity',
        'miss_distance_mKm': '🟢 Miss Distance',
        'absolute_magnitude': '🟣 Absolute Magnitude',
        'eccentricity': '🟠 Orbital Eccentricity',
        'inclination_deg': '🔴 Orbital Inclination',
    }

    feat_imp_df['Label'] = feat_imp_df['Feature'].map(feat_descriptions)

    fig_imp = go.Figure()
    fig_imp.add_trace(go.Bar(
        x=feat_imp_df['Importance'],
        y=feat_imp_df['Label'],
        orientation='h',
        marker_color=px.colors.qualitative.Set2[:len(feat_imp_df)],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Importance: %{x:.3f}<extra></extra>"
        )
    ))

    fig_imp.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Importance Score",
        yaxis=dict(tickfont=dict(size=12)),
    )

    st.plotly_chart(fig_imp, use_container_width=True)

    # Plain-language explanation of top feature
    top_feature = feat_imp_df.iloc[-1]
    st.success(
        f"🏆 **Most Important Feature:** {top_feature['Label']}\n\n"
        f"This feature has the highest influence ({top_feature['Importance']:.1%}) "
        f"on whether the model classifies an asteroid as hazardous or not."
    )

# ── Dataset explorer ──────────────────────────────────────────
st.markdown("---")
st.subheader("🔭 Dataset Explorer")

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    show_haz = st.selectbox("Filter by class",
                            ["All", "Hazardous only", "Not Hazardous only"])
with col_f2:
    sort_by = st.selectbox("Sort by", FEATURES)
with col_f3:
    n_rows = st.slider("Rows to show", 5, 50, 10)

display_df = df.copy()
if show_haz == "Hazardous only":
    display_df = display_df[display_df.is_hazardous == 1]
elif show_haz == "Not Hazardous only":
    display_df = display_df[display_df.is_hazardous == 0]

display_df = display_df.sort_values(sort_by, ascending=False).head(n_rows)
display_df['is_hazardous'] = display_df['is_hazardous'].map({1: '⚠️ Yes', 0: '✅ No'})
st.dataframe(display_df, use_container_width=True)

# ── Stats ─────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Asteroids", len(df))
c2.metric("Hazardous", df.is_hazardous.sum())
c3.metric("Not Hazardous", (df.is_hazardous == 0).sum())
c4.metric("Hazard Rate", f"{df.is_hazardous.mean():.1%}")

st.markdown("---")
st.caption("Applied AI Study Group #10 · Week 1 · Asteroid Hazard Prediction Workshop")
