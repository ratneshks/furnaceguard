import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time, os, joblib, math
from datetime import datetime

st.set_page_config(page_title="FurnaceGuard | Tata Steel", layout="wide", initial_sidebar_state="expanded")

# --- AUTO-TRAIN ---
if not os.path.exists("model.pkl"):
    with st.spinner("Training AI Model..."):
        import train_model
        df = train_model.generate_synthetic_data(10000)
        train_model.train_and_save_model(df, "model.pkl")

@st.cache_resource
def load_ml():
    return joblib.load("model.pkl")

art = load_ml()
model, scaler, feature_cols = art["model"], art["scaler"], art["features"]
model_metrics = art.get("metrics", {"mae": 0, "r2": 0})
feat_imp = art.get("feature_importance", {})

# --- CSS ---
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
#MainMenu,footer,header{display:none!important}
div[data-testid="stDecoration"],.stDeployButton{display:none}
.stApp{background:linear-gradient(145deg,#060b18 0%,#0d1526 40%,#0a1020 100%);font-family:'Inter',sans-serif}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#080e1e 0%,#0d1730 100%)!important;border-right:1px solid rgba(0,51,160,0.2)}
section[data-testid="stSidebar"] *{color:#a0b0c4!important}
.hdr{background:linear-gradient(135deg,#0033A0 0%,#001a5e 50%,#000c2a 100%);padding:24px 36px;border-radius:14px;border:1px solid rgba(77,166,255,0.2);margin-bottom:20px;box-shadow:0 6px 30px rgba(0,51,160,0.3);position:relative;overflow:hidden}
.hdr::after{content:'';position:absolute;top:-60%;right:-15%;width:350px;height:350px;background:radial-gradient(circle,rgba(77,166,255,0.12) 0%,transparent 70%);pointer-events:none}
.hdr h1{font-size:28px;font-weight:800;color:#fff;margin:0;letter-spacing:-0.5px}
.hdr p{margin:4px 0 0;font-size:12px;color:rgba(160,176,196,0.7);letter-spacing:2px;text-transform:uppercase}
.hdr .meta{font-size:11px;color:rgba(160,176,196,0.45);margin-top:6px}
.kpi{background:rgba(13,21,38,0.85);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:18px 14px;text-align:center;transition:border-color .3s}
.kpi:hover{border-color:rgba(0,51,160,0.4)}
.kpi .ic{font-size:22px;margin-bottom:4px}
.kpi .lb{font-size:10px;color:#667;text-transform:uppercase;letter-spacing:1.5px;font-weight:600}
.kpi .vl{font-size:26px;font-weight:800;color:#fff;margin:3px 0}
.kpi .un{font-size:11px;color:#556;font-weight:500}
.rul-box{background:rgba(13,21,38,0.9);border:1px solid rgba(0,51,160,0.35);border-radius:14px;padding:28px;text-align:center;box-shadow:0 6px 28px rgba(0,51,160,0.12)}
.rul-box .lb{font-size:12px;color:#889;text-transform:uppercase;letter-spacing:2px;font-weight:600}
.rul-box .vl{font-size:52px;font-weight:900;background:linear-gradient(135deg,#4da6ff,#0033A0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:6px 0;line-height:1}
.rul-box .un{font-size:14px;color:#667;font-weight:500}
.st-g{background:linear-gradient(135deg,rgba(30,142,62,0.12),rgba(30,142,62,0.04));border:1px solid rgba(30,142,62,0.45);border-radius:12px;padding:20px;text-align:center}
.st-g .dot{display:inline-block;width:9px;height:9px;background:#1e8e3e;border-radius:50%;margin-right:7px;box-shadow:0 0 10px rgba(30,142,62,0.6);animation:pg 2s infinite}
.st-g .tx{font-size:16px;font-weight:700;color:#34d058}
.st-g .su{font-size:11px;color:#1e8e3e;margin-top:6px}
@keyframes pg{0%,100%{box-shadow:0 0 6px rgba(30,142,62,0.3)}50%{box-shadow:0 0 18px rgba(30,142,62,0.7)}}
.st-a{background:linear-gradient(135deg,rgba(255,191,0,0.12),rgba(255,191,0,0.04));border:1px solid rgba(255,191,0,0.45);border-radius:12px;padding:20px;text-align:center}
.st-a .dot{display:inline-block;width:9px;height:9px;background:#FFBF00;border-radius:50%;margin-right:7px;box-shadow:0 0 10px rgba(255,191,0,0.6);animation:pa 1.5s infinite}
.st-a .tx{font-size:16px;font-weight:700;color:#FFBF00}
.st-a .su{font-size:11px;color:#b08c00;margin-top:6px}
@keyframes pa{0%,100%{box-shadow:0 0 6px rgba(255,191,0,0.3)}50%{box-shadow:0 0 18px rgba(255,191,0,0.7)}}
.st-r{background:linear-gradient(135deg,rgba(227,24,55,0.15),rgba(227,24,55,0.04));border:1px solid rgba(227,24,55,0.55);border-radius:12px;padding:20px;text-align:center;animation:fr 1s infinite}
.st-r .dot{display:inline-block;width:9px;height:9px;background:#E31837;border-radius:50%;margin-right:7px;box-shadow:0 0 14px rgba(227,24,55,0.8)}
.st-r .tx{font-size:16px;font-weight:700;color:#ff4d6a}
.st-r .su{font-size:11px;color:#ff4d6a;margin-top:6px}
@keyframes fr{0%,100%{opacity:1}50%{opacity:.65}}
.sec{font-size:12px;font-weight:700;color:#778;text-transform:uppercase;letter-spacing:2px;margin:24px 0 12px;padding-bottom:6px;border-bottom:1px solid rgba(255,255,255,0.05)}
.gc{background:rgba(13,21,38,0.75);border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:20px;box-shadow:0 4px 20px rgba(0,0,0,0.25)}
.sb{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);border-radius:8px;padding:12px;margin-bottom:10px}
.sb .sl{font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:#556!important;font-weight:600}
.sb .sv{font-size:14px;font-weight:700;color:#fff!important}
.lt{width:100%;border-collapse:separate;border-spacing:0 5px}
.lt th{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#556;padding:6px 10px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.05)}
.lt td{font-size:12px;color:#a0b0c4;padding:8px 10px;background:rgba(255,255,255,0.02)}
.lt tr td:first-child{border-radius:6px 0 0 6px}
.lt tr td:last-child{border-radius:0 6px 6px 0}
.br{display:inline-block;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700}
.br-r{background:rgba(227,24,55,0.2);color:#ff4d6a}
.br-a{background:rgba(255,191,0,0.2);color:#FFBF00}
.roi{background:linear-gradient(135deg,rgba(30,142,62,0.1),rgba(0,51,160,0.08));border:1px solid rgba(30,142,62,0.3);border-radius:12px;padding:18px;text-align:center}
.roi .rl{font-size:10px;color:#778;text-transform:uppercase;letter-spacing:1.5px;font-weight:600}
.roi .rv{font-size:28px;font-weight:800;color:#34d058;margin:4px 0}
.roi .rs{font-size:11px;color:#556}
.rec{background:rgba(13,21,38,0.8);border-left:3px solid #4da6ff;border-radius:0 10px 10px 0;padding:14px 16px;margin-top:8px}
.rec .rt{font-size:12px;font-weight:700;color:#c8d6e5}
.rec .rd{font-size:11px;color:#889;margin-top:4px}
</style>""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""<div style="text-align:center;padding:12px 0 20px">
        <div style="font-size:24px;font-weight:900;color:#4da6ff!important;letter-spacing:-1px">⚙️ FurnaceGuard</div>
        <div style="font-size:9px;color:#556!important;letter-spacing:2px;text-transform:uppercase;margin-top:3px">AI Predictive Maintenance</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div class="sb"><div class="sl">Plant</div><div class="sv">Tata Steel — Jamshedpur</div></div>
    <div class="sb"><div class="sl">Mill Line</div><div class="sv">Tandem Cold Rolling Mill #3</div></div>
    <div class="sb"><div class="sl">Current Shift</div><div class="sv">Shift B — 14:00 to 22:00</div></div>
    <div class="sb"><div class="sl">Shift Engineer</div><div class="sv">Rajesh K. Verma</div></div>
    <div class="sb"><div class="sl">AI Model</div><div class="sv">Random Forest v1.0</div></div>
    <div class="sb"><div class="sl">Model Accuracy (R²)</div><div class="sv">{:.1f}%</div></div>
    <div class="sb"><div class="sl">Prediction Error (MAE)</div><div class="sv">{:.1f} hours</div></div>
    """.format(model_metrics["r2"]*100, model_metrics["mae"]), unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="sl" style="margin-bottom:6px;font-size:10px;color:#556!important;letter-spacing:1.5px;text-transform:uppercase;font-weight:600">⚠️ DANGER THRESHOLDS</div>', unsafe_allow_html=True)
    DANGER_VIB = st.slider("Vibration (mm/s)", 5.0, 25.0, 15.0)
    DANGER_TEMP = st.slider("Temperature (°C)", 60.0, 150.0, 100.0)

# --- HEADER ---
now_s = datetime.now().strftime("%d %b %Y  •  %H:%M:%S IST")
st.markdown(f'<div class="hdr"><h1>FurnaceGuard: AI Predictive Maintenance</h1><p>Tandem Cold Mill — Live Telemetry Dashboard</p><div class="meta">🕐 {now_s}  •  Tata Steel, Jamshedpur  •  IoT Edge Gateway Active</div></div>', unsafe_allow_html=True)

# --- PLACEHOLDERS ---
st.markdown('<div class="sec">📡 Live Sensor Readings</div>', unsafe_allow_html=True)
kpi_row = st.columns(4)
kpi_ph = [c.empty() for c in kpi_row]

st.markdown('<div class="sec">🧠 AI Prediction & System Status</div>', unsafe_allow_html=True)
ai_row = st.columns([1.1, 1.4, 0.9, 0.9])
gauge_ph, rul_ph, status_ph, roi_ph = [c.empty() for c in ai_row]

st.markdown('<div class="sec">📈 Live Sensor Feeds — Trailing 50 Seconds</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
vib_ph, temp_ph = c1.empty(), c2.empty()
c3, c4 = st.columns(2)
ac_ph, load_ph = c3.empty(), c4.empty()

st.markdown('<div class="sec">🔧 Maintenance Recommendation</div>', unsafe_allow_html=True)
rec_ph = st.empty()

st.markdown('<div class="sec">📋 Alert Action Log</div>', unsafe_allow_html=True)
log_ph = st.empty()

# --- STATE ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Timestamp"]+feature_cols)
if 'action_log' not in st.session_state:
    st.session_state.action_log = []
if 'alert_count' not in st.session_state:
    st.session_state.alert_count = {"amber": 0, "red": 0}
if 'last_alert_level' not in st.session_state:
    st.session_state.last_alert_level = "green"
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()

from data_stream import get_realtime_data_generator
data_gen = get_realtime_data_generator()

CL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,16,32,0.6)",
    font=dict(family="Inter", color="#778", size=10),
    margin=dict(l=45, r=15, t=35, b=35), height=280,
    xaxis=dict(gridcolor="rgba(255,255,255,0.03)", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.03)", showgrid=True, zeroline=False))

COST_PER_HOUR_DOWNTIME = 8.5  # ₹ Lakhs per hour of unplanned downtime

# --- LOOP ---
for tick in range(2000):
    r = next(data_gen)
    row = {k: r[k] for k in ["Timestamp"] + feature_cols}
    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([row])], ignore_index=True).tail(50)
    df_h = st.session_state.history

    X = pd.DataFrame([r])[feature_cols]
    rul = max(0.0, float(model.predict(scaler.transform(X))[0]))
    health = min(100.0, max(0.0, (rul / 250.0) * 100))

    # KPIs
    kd = [("📳","Vibration",f"{r['Vibration_Velocity_mm_s']:.2f}","mm/s"),
          ("🌡️","Motor Temp",f"{r['Motor_Temperature_C']:.1f}","°C"),
          ("🔊","Acoustic",f"{r['Acoustic_Frequency_Hz']:.0f}","Hz"),
          ("⚖️","Roll Load",f"{r['Load_Tonnage']:.1f}","Tonnes")]
    for i,(ic,lb,vl,un) in enumerate(kd):
        kpi_ph[i].markdown(f'<div class="kpi"><div class="ic">{ic}</div><div class="lb">{lb}</div><div class="vl">{vl}</div><div class="un">{un}</div></div>', unsafe_allow_html=True)

    # Gauge
    gc = "#1e8e3e" if health >= 70 else "#FFBF00" if health >= 30 else "#E31837"
    fg = go.Figure(go.Indicator(mode="gauge+number", value=health,
        number=dict(suffix="%", font=dict(size=36, color="#fff", family="Inter")),
        title=dict(text="Equipment Health", font=dict(size=12, color="#889", family="Inter")),
        gauge=dict(axis=dict(range=[0,100], tickcolor="#334", tickwidth=1, dtick=25),
            bar=dict(color=gc, thickness=0.28), bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0,30],color="rgba(227,24,55,0.08)"),
                   dict(range=[30,70],color="rgba(255,191,0,0.06)"),
                   dict(range=[70,100],color="rgba(30,142,62,0.06)")],
            threshold=dict(line=dict(color="#E31837",width=2),thickness=0.75,value=20))))
    fg.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter",color="#889"), height=240, margin=dict(l=25,r=25,t=55,b=15))
    gauge_ph.plotly_chart(fg, use_container_width=True)

    # RUL
    rul_ph.markdown(f'<div class="rul-box"><div class="lb">Predicted Remaining Useful Life</div><div class="vl">{rul:.1f}</div><div class="un">Hours until maintenance required</div></div>', unsafe_allow_html=True)

    # Status
    current_level = "green" if rul > 72 else "amber" if rul >= 24 else "red"
    if current_level == "green":
        status_ph.markdown('<div class="st-g"><div><span class="dot"></span><span class="tx">All Systems Nominal</span></div><div class="su">No action required</div></div>', unsafe_allow_html=True)
    elif current_level == "amber":
        status_ph.markdown('<div class="st-a"><div><span class="dot"></span><span class="tx">Maintenance Soon</span></div><div class="su">Schedule inspection within 24h</div></div>', unsafe_allow_html=True)
        if st.session_state.last_alert_level != "amber":
            st.session_state.action_log.insert(0, {"Time": datetime.now().strftime("%H:%M:%S"), "Level": "Amber", "Detail": f"RUL dropped to {rul:.1f}h — schedule maintenance"})
            st.session_state.alert_count["amber"] += 1
    else:
        status_ph.markdown('<div class="st-r"><div><span class="dot"></span><span class="tx">CRITICAL ALERT</span></div><div class="su">Immediate shutdown recommended</div></div>', unsafe_allow_html=True)
        if st.session_state.last_alert_level != "red":
            st.session_state.action_log.insert(0, {"Time": datetime.now().strftime("%H:%M:%S"), "Level": "Red", "Detail": f"CRITICAL — RUL is {rul:.1f}h — stop mill immediately"})
            st.session_state.alert_count["red"] += 1
    st.session_state.last_alert_level = current_level

    # ROI Card
    uptime_min = (datetime.now() - st.session_state.start_time).total_seconds() / 60
    avoided = st.session_state.alert_count["red"]
    savings = avoided * 2.0 * COST_PER_HOUR_DOWNTIME
    roi_ph.markdown(f'<div class="roi"><div class="rl">Est. Downtime Savings</div><div class="rv">₹{savings:.1f}L</div><div class="rs">{avoided} critical failures predicted<br>Session: {uptime_min:.0f} min</div></div>', unsafe_allow_html=True)

    # Charts
    def make_chart(df, col, title, color, fill_color, danger_val):
        f = go.Figure()
        f.add_trace(go.Scatter(x=df["Timestamp"], y=df[col], mode="lines", name=title,
            line=dict(color=color, width=2.2), fill="tozeroy", fillcolor=fill_color))
        f.add_hline(y=danger_val, line_dash="dash", line_color="#E31837", line_width=1.2,
            annotation_text="Danger", annotation_font_color="#E31837", annotation_font_size=10)
        f.update_layout(title=dict(text=title, font=dict(size=12, color="#a0b0c4")), showlegend=False, **CL)
        return f

    vib_ph.plotly_chart(make_chart(df_h, "Vibration_Velocity_mm_s", "Vibration Velocity (mm/s)", "#4da6ff", "rgba(77,166,255,0.06)", DANGER_VIB), use_container_width=True)
    temp_ph.plotly_chart(make_chart(df_h, "Motor_Temperature_C", "Motor Temperature (°C)", "#FFBF00", "rgba(255,191,0,0.05)", DANGER_TEMP), use_container_width=True)

    fac = go.Figure()
    fac.add_trace(go.Scatter(x=df_h["Timestamp"], y=df_h["Acoustic_Frequency_Hz"], mode="lines",
        line=dict(color="#a855f7", width=2.2), fill="tozeroy", fillcolor="rgba(168,85,247,0.05)"))
    fac.update_layout(title=dict(text="Acoustic Frequency (Hz)", font=dict(size=12, color="#a0b0c4")), showlegend=False, **CL)
    ac_ph.plotly_chart(fac, use_container_width=True)

    fld = go.Figure()
    fld.add_trace(go.Scatter(x=df_h["Timestamp"], y=df_h["Load_Tonnage"], mode="lines",
        line=dict(color="#06b6d4", width=2.2), fill="tozeroy", fillcolor="rgba(6,182,212,0.05)"))
    fld.update_layout(title=dict(text="Roll Load (Tonnes)", font=dict(size=12, color="#a0b0c4")), showlegend=False, **CL)
    load_ph.plotly_chart(fld, use_container_width=True)

    # Maintenance Recommendation
    if rul > 150:
        rec_ph.markdown('<div class="rec"><div class="rt">✅ Continue Normal Operation</div><div class="rd">All parameters within ISO 10816 limits. Next scheduled inspection: per maintenance calendar.</div></div>', unsafe_allow_html=True)
    elif rul > 72:
        rec_ph.markdown('<div class="rec"><div class="rt">📋 Plan Preventive Maintenance</div><div class="rd">Bearing vibration trending upward. Order replacement bearings (SKF 6310-2RS). Schedule lubrication check within 48 hours.</div></div>', unsafe_allow_html=True)
    elif rul >= 24:
        rec_ph.markdown(f'<div class="rec" style="border-color:#FFBF00"><div class="rt" style="color:#FFBF00">⚠️ Schedule Urgent Maintenance — {rul:.0f}h Remaining</div><div class="rd">Motor temperature elevated. Inspect drive motor bearings, check coolant flow rate, and verify roll alignment. Assign maintenance crew for next shift changeover.</div></div>', unsafe_allow_html=True)
    else:
        rec_ph.markdown(f'<div class="rec" style="border-color:#E31837"><div class="rt" style="color:#ff4d6a">🚨 EMERGENCY — Initiate Controlled Shutdown ({rul:.0f}h RUL)</div><div class="rd">Multiple sensor thresholds breached. Execute Emergency Procedure EP-TCM-003: Reduce rolling speed to 50%, alert floor supervisor, begin controlled mill stop sequence. Dispatch maintenance team immediately.</div></div>', unsafe_allow_html=True)

    # Action Log
    logs = st.session_state.action_log[:12]
    if logs:
        rows = ""
        for lg in logs:
            bc = "br-r" if lg["Level"] == "Red" else "br-a"
            rows += f'<tr><td>{lg["Time"]}</td><td><span class="br {bc}">{lg["Level"]}</span></td><td>{lg["Detail"]}</td></tr>'
        log_ph.markdown(f'<div class="gc"><table class="lt"><thead><tr><th>Time</th><th>Level</th><th>Detail</th></tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)
    else:
        log_ph.markdown('<div class="gc" style="text-align:center;padding:24px;color:#556">✅ No alerts — all systems operating within safe parameters</div>', unsafe_allow_html=True)

    time.sleep(1)
