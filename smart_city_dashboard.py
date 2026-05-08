import streamlit as st
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="UrbanPulse",
    page_icon="🌆",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Main Background */

.stApp {
    background:
    linear-gradient(
        135deg,
        #0f0c29 0%,
        #302b63 35%,
        #24243e 70%,
        #000428 100%
    );

    color: white;
}

/* Main Title */

.main-title {

    font-size: 72px;

    font-weight: 900;

    text-align: center;

    margin-top: 20px;

    background: linear-gradient(
        90deg,
        #ff00cc,
        #00ffff,
        #00ff99,
        #ffff00,
        #ff6600
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

/* Subtitle */

.sub-title {

    text-align: center;

    font-size: 24px;

    color: #d1d5db;

    margin-bottom: 40px;

    font-weight: bold;
}

/* Sidebar */

section[data-testid="stSidebar"] {

    background:
    linear-gradient(
        180deg,
        #141e30,
        #243b55
    );
}

/* Buttons */

.stButton > button {

    width: 100%;

    border-radius: 18px;

    border: none;

    padding: 15px;

    font-size: 20px;

    font-weight: bold;

    color: white;

    background:
    linear-gradient(
        90deg,
        #ff00cc,
        #3333ff,
        #00ffff
    );
}

/* Metric Cards */

.metric-card {

    background: rgba(255,255,255,0.08);

    backdrop-filter: blur(20px);

    border-radius: 25px;

    padding: 30px;

    text-align: center;

    border: 1px solid rgba(255,255,255,0.1);

    margin-bottom: 20px;
}

.metric-card h2 {

    color: #00ffff;
}

.metric-card h1 {

    font-size: 42px;

    font-weight: 900;
}

/* Recommendation Cards */

.recommend-card {

    background: rgba(255,255,255,0.08);

    border-left: 6px solid #00ffff;

    padding: 20px;

    border-radius: 20px;

    margin-bottom: 15px;

    font-size: 18px;
}

/* Footer */

.footer {

    text-align: center;

    margin-top: 50px;

    color: #d1d5db;

    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS
# =========================================================

carbon_model = pickle.load(open("carbon_model.pkl", "rb"))
activity_model = pickle.load(open("activity_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# =========================================================
# HEADER
# =========================================================

st.markdown(
    "<div class='main-title'>🌆 UrbanPulse</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Smart City Citizen Analytics Dashboard</div>",
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌆 Navigation")

menu = st.sidebar.radio(
    "Choose Section",
    [
        "Citizen Input",
        "Predictions",
        "Analytics",
        "Recommendations"
    ]
)

# =========================================================
# SESSION STATE
# =========================================================

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

# =========================================================
# INPUT PAGE
# =========================================================

if menu == "Citizen Input":

    st.header("🧍 Citizen Lifestyle Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input("Age", 18, 80, 25)

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        transport = st.selectbox(
            "Mode of Transport",
            ["Bike", "Bus", "Car", "Metro", "Walking"]
        )

        work_hours = st.slider(
            "Work Hours",
            0,
            15,
            8
        )

        shopping_hours = st.slider(
            "Shopping Hours",
            0,
            10,
            2
        )

        entertainment_hours = st.slider(
            "Entertainment Hours",
            0,
            10,
            2
        )

        sleep_hours = st.slider(
            "Sleep Hours",
            0,
            12,
            7
        )

    with col2:

        energy = st.number_input(
            "Home Energy Consumption (kWh)",
            0.0,
            1000.0,
            250.0
        )

        charging = st.slider(
            "Charging Station Usage",
            0,
            20,
            2
        )

        steps = st.number_input(
            "Steps Walked",
            0,
            30000,
            7000
        )

        calories = st.number_input(
            "Calories Burned",
            0,
            5000,
            500
        )

        social_media = st.slider(
            "Social Media Hours",
            0,
            15,
            4
        )

        public_events = st.slider(
            "Public Events Hours",
            0,
            15,
            1
        )

    # =====================================================
    # BUTTON
    # =====================================================

    if st.button("🚀 Run Smart Analysis"):

        gender_code = encoders["gender"].transform([gender])[0]

        transport_code = encoders["transport"].transform([transport])[0]

        input_data = np.array([[
            age,
            gender_code,
            transport_code,
            work_hours,
            shopping_hours,
            entertainment_hours,
            energy,
            charging,
            steps,
            calories,
            sleep_hours,
            social_media,
            public_events
        ]])

        scaled_data = scaler.transform(input_data)

        # =====================================================
        # PREDICTIONS
        # =====================================================

        carbon_prediction = carbon_model.predict(scaled_data)[0]

        activity_prediction = activity_model.predict(scaled_data)[0]

        activity_label = encoders["activity"].inverse_transform(
            [activity_prediction]
        )[0]

        sustainability_score = max(
            0,
            100 - (carbon_prediction / 10)
        )

        health_score = min(
            100,
            (steps / 100) + (sleep_hours * 5)
        )

        energy_efficiency = max(
            0,
            100 - (energy / 10)
        )

        lifestyle_score = (
            sustainability_score +
            health_score +
            energy_efficiency
        ) / 3

        st.session_state.results = {

            "carbon": carbon_prediction,

            "activity": activity_label,

            "score": sustainability_score,

            "steps": steps,

            "sleep": sleep_hours,

            "energy": energy,

            "health_score": health_score,

            "energy_efficiency": energy_efficiency,

            "lifestyle_score": lifestyle_score,

            "age": age
        }

        st.session_state.prediction_done = True

        st.success("✅ Analysis Completed Successfully!")

# =========================================================
# PREDICTIONS PAGE
# =========================================================

elif menu == "Predictions":

    st.header("🔮 Smart Predictions")

    if st.session_state.prediction_done:

        results = st.session_state.results

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(f"""
            <div class='metric-card'>
            <h2>🌍 Carbon Footprint</h2>
            <h1>{results['carbon']:.2f}</h1>
            <p>kgCO2 Emission</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown(f"""
            <div class='metric-card'>
            <h2>🚶 Activity Level</h2>
            <h1>{results['activity']}</h1>
            <p>Citizen Health Index</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:

            st.markdown(f"""
            <div class='metric-card'>
            <h2>♻ Sustainability</h2>
            <h1>{results['score']:.0f}/100</h1>
            <p>Eco Smart Score</p>
            </div>
            """, unsafe_allow_html=True)

        # =====================================================
        # PIE CHART
        # =====================================================

        pie = go.Figure(
            data=[go.Pie(

                labels=[
                    "Health",
                    "Energy",
                    "Sustainability",
                    "Lifestyle"
                ],

                values=[
                    results["health_score"],
                    results["energy_efficiency"],
                    results["score"],
                    results["lifestyle_score"]
                ],

                hole=0.45,

                textinfo='label+percent'
            )]
        )

        pie.update_layout(

            title="🌟 Citizen Smart Performance Distribution",

            template="plotly_dark",

            height=600,

            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(pie, use_container_width=True)

        # =====================================================
        # AGE GROUP COMPARISON GRAPH
        # =====================================================

        st.markdown("## 👥 Age Group Comparison Analytics")

        categories = [
            "Health",
            "Energy",
            "Sustainability",
            "Lifestyle"
        ]

        user_values = [

            results["health_score"],

            results["energy_efficiency"],

            results["score"],

            results["lifestyle_score"]
        ]

        age_18_24 = [72, 68, 70, 71]

        age_25_39 = [65, 62, 66, 64]

        age_40_59 = [58, 60, 63, 59]

        age_60_plus = [52, 57, 61, 55]

        user_age = results["age"]

        comparison_fig = go.Figure()

        comparison_fig.add_trace(go.Scatter(

            x=categories,

            y=user_values,

            mode='lines+markers',

            name=f'Current User (Age {user_age})',

            line=dict(width=6)
        ))

        comparison_fig.add_trace(go.Scatter(

            x=categories,

            y=age_18_24,

            mode='lines+markers',

            name='Age Group 18-24',

            line=dict(width=3, dash='dash')
        ))

        comparison_fig.add_trace(go.Scatter(

            x=categories,

            y=age_25_39,

            mode='lines+markers',

            name='Age Group 25-39',

            line=dict(width=3, dash='dot')
        ))

        comparison_fig.add_trace(go.Scatter(

            x=categories,

            y=age_40_59,

            mode='lines+markers',

            name='Age Group 40-59',

            line=dict(width=3)
        ))

        comparison_fig.add_trace(go.Scatter(

            x=categories,

            y=age_60_plus,

            mode='lines+markers',

            name='Age Group 60+',

            line=dict(width=3, dash='longdash')
        ))

        comparison_fig.update_layout(

            title="📊 Smart Citizen Age Group Comparison",

            title_x=0.25,

            template="plotly_dark",

            height=700,

            yaxis=dict(range=[0, 100]),

            paper_bgcolor='rgba(0,0,0,0)',

            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(
            comparison_fig,
            use_container_width=True
        )

    else:

        st.warning("⚠ Please complete Citizen Input first.")

# =========================================================
# ANALYTICS PAGE
# =========================================================

elif menu == "Analytics":

    st.header("📊 Smart Performance Analytics")

    if st.session_state.prediction_done:

        results = st.session_state.results

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.markdown(f"""
            <div class='metric-card'>
            <h2>💪 Health</h2>
            <h1>{results["health_score"]:.0f}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown(f"""
            <div class='metric-card'>
            <h2>⚡ Energy</h2>
            <h1>{results["energy_efficiency"]:.0f}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col3:

            st.markdown(f"""
            <div class='metric-card'>
            <h2>🌱 Sustainability</h2>
            <h1>{results["score"]:.0f}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col4:

            st.markdown(f"""
            <div class='metric-card'>
            <h2>🏆 Lifestyle</h2>
            <h1>{results["lifestyle_score"]:.0f}</h1>
            </div>
            """, unsafe_allow_html=True)

        categories = [
            "Health",
            "Energy",
            "Sustainability",
            "Lifestyle"
        ]

        values = [
            results["health_score"],
            results["energy_efficiency"],
            results["score"],
            results["lifestyle_score"]
        ]

        polar_chart = go.Figure()

        polar_chart.add_trace(go.Scatterpolar(

            r=values,

            theta=categories,

            fill='toself',

            name='Citizen Analysis',

            line=dict(width=4)
        ))

        polar_chart.update_layout(

            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),

            template="plotly_dark",

            title="🌌 Smart Citizen Polar Analysis",

            title_x=0.25,

            height=650,

            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(polar_chart, use_container_width=True)

        chart_data = {
            "Metrics": categories,
            "Scores": values
        }

        area_fig = px.area(

            chart_data,

            x="Metrics",

            y="Scores",

            markers=True,

            template="plotly_dark",

            height=500
        )

        area_fig.update_layout(

            title="📈 Smart Lifestyle Performance",

            title_x=0.3,

            paper_bgcolor='rgba(0,0,0,0)',

            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(area_fig, use_container_width=True)

    else:

        st.warning("⚠ Please run analysis first.")

# =========================================================
# RECOMMENDATIONS PAGE
# =========================================================

elif menu == "Recommendations":

    st.header("💡 Smart Recommendations")

    if st.session_state.prediction_done:

        results = st.session_state.results

        if results["carbon"] > 500:

            st.markdown("""
            <div class='recommend-card'>
            🌱 Use public transport or cycling more frequently.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='recommend-card'>
            ⚡ Reduce unnecessary appliance usage during peak hours.
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class='recommend-card'>
            ✅ Your environmental impact is maintained efficiently.
            </div>
            """, unsafe_allow_html=True)

        if results["steps"] < 5000:

            st.markdown("""
            <div class='recommend-card'>
            🚶 Try achieving at least 8,000 daily steps.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='recommend-card'>
            🏃 Include light workouts or evening walks regularly.
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class='recommend-card'>
            💪 Your physical activity level is excellent.
            </div>
            """, unsafe_allow_html=True)

        if results["sleep"] < 6:

            st.markdown("""
            <div class='recommend-card'>
            😴 Improve sleep quality by maintaining a fixed bedtime.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='recommend-card'>
            📵 Reduce mobile usage before sleeping.
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class='recommend-card'>
            🌙 Your sleep schedule appears balanced.
            </div>
            """, unsafe_allow_html=True)

        if results["energy"] > 300:

            st.markdown("""
            <div class='recommend-card'>
            ⚡ Switch to energy-efficient lighting systems.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='recommend-card'>
            🔌 Disconnect idle electronic devices to save power.
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class='recommend-card'>
            🔋 Your energy consumption is within efficient limits.
            </div>
            """, unsafe_allow_html=True)

        if results["lifestyle_score"] > 75:

            st.markdown("""
            <div class='recommend-card'>
            🏆 Excellent smart lifestyle performance detected.
            </div>
            """, unsafe_allow_html=True)

        elif results["lifestyle_score"] > 50:

            st.markdown("""
            <div class='recommend-card'>
            📈 Moderate lifestyle quality. Small improvements can boost performance.
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class='recommend-card'>
            🚨 Lifestyle metrics indicate improvement opportunities in health and sustainability.
            </div>
            """, unsafe_allow_html=True)

    else:

        st.warning("⚠ Run analysis first.")

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class='footer'>
🌆 UrbanPulse • Smart City Intelligence System • 2026
</div>
""", unsafe_allow_html=True)