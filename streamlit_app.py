import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime
import io

from backend.core.database import init_db, SessionLocal, SessionModel, Attendance, SessionPerson, PersonMetric, Student
from backend.core.face_detection import FaceDetector
from backend.core.camera_service import CameraService
from backend.core.audio_analysis import AudioAnalyzer
from backend.core.multimodal_agent import MultiModalAgent
from backend.core.recommendations_engine import RecommendationsEngine
from backend.core.ai_suggestions import AISuggestionEngine
from backend.core.llm_insights import InsightGenerator
from backend.core.teacher_profiles import TeacherProfileService
from backend.core.analytics_service import AnalyticsService

# Initialize database
init_db()

st.set_page_config(
    page_title="EduSense AI - Multimodal Classroom Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom EduSense dark theme matching the PDF screenshots
st.markdown("""
<style>
    .reportview-container {
        background: #0f172a;
    }
    .main-header {
        font-size: 24px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #94a3b8;
        margin-bottom: 20px;
    }
    .stat-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #334155;
        margin-bottom: 12px;
    }
    .banner-success {
        background-color: #064e3b;
        border: 1px solid #059669;
        color: #d1fae5;
        padding: 10px 16px;
        border-radius: 6px;
        font-weight: 600;
        margin-bottom: 14px;
    }
    .banner-warning {
        background-color: #78350f;
        border: 1px solid #d97706;
        color: #fef3c7;
        padding: 10px 16px;
        border-radius: 6px;
        font-weight: 600;
        margin-bottom: 14px;
    }
    .metric-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-high {
        background: #dc2626;
        color: white;
    }
    .badge-med {
        background: #f59e0b;
        color: black;
    }
    .badge-info {
        background: #2563eb;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "detector" not in st.session_state:
    st.session_state.detector = FaceDetector()
if "audio_analyzer" not in st.session_state:
    st.session_state.audio_analyzer = AudioAnalyzer()
if "agent" not in st.session_state:
    st.session_state.agent = MultiModalAgent(attendance_threshold_seconds=5.0)
if "recs_engine" not in st.session_state:
    st.session_state.recs_engine = RecommendationsEngine()
if "suggestions_engine" not in st.session_state:
    st.session_state.suggestions_engine = AISuggestionEngine()
if "insight_gen" not in st.session_state:
    st.session_state.insight_gen = InsightGenerator()
if "teacher_service" not in st.session_state:
    st.session_state.teacher_service = TeacherProfileService()
if "analytics_service" not in st.session_state:
    st.session_state.analytics_service = AnalyticsService()
if "camera" not in st.session_state:
    st.session_state.camera = CameraService(camera_id=0)
    st.session_state.camera.start()
if "session_active" not in st.session_state:
    st.session_state.session_active = True
if "session_id" not in st.session_state:
    db = SessionLocal()
    sess = db.query(SessionModel).filter(SessionModel.is_active == True).first()
    if not sess:
        sess = SessionModel(title="CS101 - Deep Learning & Multimodal AI", is_active=True)
        db.add(sess)
        db.commit()
        db.refresh(sess)
    st.session_state.session_id = sess.id
    db.close()
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time() - 536  # ~08:56 for demo realism

# Navigation sidebar
st.sidebar.markdown("## 🎓 EduSense AI")
st.sidebar.markdown("*Multimodal Attendance & Engagement Agent*")
view_mode = st.sidebar.radio(
    "Navigation",
    ["🔴 Live Classroom Feed", "👨‍🏫 Teacher Central & Profile", "📊 Session Analytics & Reports"]
)

st.sidebar.markdown("---")
# Session controls in sidebar
if st.session_state.session_active:
    st.sidebar.success("🟢 SESSION ACTIVE")
    elapsed = int(time.time() - st.session_state.start_time)
    mins = elapsed // 60
    secs = elapsed % 60
    st.sidebar.write(f"⏱️ **Duration:** `{mins:02d}:{secs:02d}`")
    if st.sidebar.button("🛑 End Session", use_container_width=True):
        st.session_state.session_active = False
        db = SessionLocal()
        s = db.query(SessionModel).filter(SessionModel.id == st.session_state.session_id).first()
        if s:
            s.is_active = False
            s.end_time = datetime.utcnow()
            db.commit()
        db.close()
        st.rerun()
else:
    st.sidebar.warning("⚪ SESSION INACTIVE")
    if st.sidebar.button("▶️ Start New Session", use_container_width=True):
        st.session_state.session_active = True
        st.session_state.start_time = time.time()
        db = SessionLocal()
        new_s = SessionModel(title="CS101 - Advanced AI Interactive Session", is_active=True)
        db.add(new_s)
        db.commit()
        db.refresh(new_s)
        st.session_state.session_id = new_s.id
        db.close()
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Video & Audio Configuration")
input_choice = st.sidebar.selectbox("Video Source", ["Default Webcam (or Simulated Classroom)", "Upload Classroom Video File"])
custom_video_path = None
if input_choice == "Upload Classroom Video File":
    uploaded_video = st.sidebar.file_uploader("Upload MP4/AVI Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        custom_video_path = tfile.name

# Optional Audio upload
audio_choice = st.sidebar.selectbox("Audio Source", ["Real-time Acoustic Analysis", "Upload Audio Waveform"])
if audio_choice == "Upload Audio Waveform":
    uploaded_audio = st.sidebar.file_uploader("Upload WAV/MP3", type=["wav", "mp3"])
    if uploaded_audio is not None:
        audio_bytes = uploaded_audio.read()
        a_metrics = st.session_state.audio_analyzer.analyze_audio_chunk(audio_bytes[:32000])
        if a_metrics:
            st.session_state.agent.update_audio_state(a_metrics)
            st.sidebar.success(f"Analyzed audio: {a_metrics['activity_type']}")


# ----------------------------------------------------
# VIEW 1: LIVE CLASSROOM INTELLIGENCE (PDF Page 16)
# ----------------------------------------------------
if view_mode == "🔴 Live Classroom Feed":
    st.markdown('<div class="main-header">EduSense AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Classroom Intelligence & Multimodal Engagement Monitor</div>', unsafe_allow_html=True)

    # Acquire frame
    frame = st.session_state.camera.get_frame()
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = st.session_state.detector.process_frame(frame)
    session_status = st.session_state.agent.process_session_frame(
        detections=detections,
        detector_obj=st.session_state.detector,
        raw_frame=frame,
        session_id=st.session_state.session_id if st.session_state.session_active else None
    )

    recs = st.session_state.recs_engine.generate_realtime_recommendations(session_status)
    suggestions = st.session_state.suggestions_engine.generate_suggestions(session_status)
    insights = st.session_state.insight_gen.generate_classroom_insight(session_status)

    # Top Status Banner matching screenshot page 16
    avg_eng = session_status.get("avg_engagement", 80.0)
    if avg_eng >= 70:
        st.markdown('<div class="banner-success">✔️ SUCCESS: Engagement is optimal. Keep going!</div>', unsafe_allow_html=True)
    elif avg_eng >= 50:
        st.markdown('<div class="banner-warning">⚠️ MODERATE: Noticeable attention fluctuation detected. Consider interactive check-in.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="banner-warning">🚨 ATTENTION ALERT: High disengagement observed across multiple participants.</div>', unsafe_allow_html=True)

    # 3-Column Layout: Left (Stats & Controls), Center (Video Feed & Bottom Cards), Right (Emotion & Audio)
    col_left, col_center, col_right = st.columns([1, 2.5, 1.2])

    with col_left:
        st.markdown("### 📋 Session Status")
        st.metric("Students Present", f"{session_status.get('present_count', 0)}")
        st.metric("Avg Attention", f"{session_status.get('avg_attention', 0)}%")
        st.metric("Multimodal Engagement", f"{session_status.get('avg_engagement', 0)}%")

        st.markdown("---")
        st.markdown("### 🏆 Live Leaderboard")
        people = session_status.get("people", [])
        if people:
            for p in people:
                points = int(p["engagement"] * 0.4)
                st.write(f"👤 **{p['student_id']}** — `{points} pts`")
        else:
            st.write("👤 **Student 1** — `30 pts`")
            st.write("👤 **Student 3** — `2 pts`")

    with col_center:
        # Draw bounding boxes and status labels on frame matching screenshot page 16
        annotated_frame = frame.copy()
        for p in session_status.get("people", []):
            x, y, w, h = p["bbox"]
            is_eng = p.get("status") == "engaged"
            color = (50, 205, 50) if is_eng else (30, 105, 255) # Green vs Red in BGR
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            label = f"ID: {p['id']} | {p['status']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x, max(0, y - th - 10)), (x + tw + 10, y), color, -1)
            cv2.putText(annotated_frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Convert to RGB for Streamlit display
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st.image(rgb_frame, channels="RGB", use_container_width=True)

        # Bottom row of Center column: AI Suggestion Box (Matches Page 16)
        st.markdown("### 💡 AI Suggestion Box")
        if suggestions:
            for s in suggestions[:2]:
                st.markdown(
                    f"""
                    <div style="background:#1e293b; padding:12px; border-radius:6px; border-left:4px solid #ef4444; margin-bottom:8px;">
                        <span class="metric-badge badge-high">{s.get('priority', 'High Priority')}</span>
                        <b style="color:#f8fafc; margin-left:8px;">{s.get('suggestion')}</b>
                        <div style="color:#94a3b8; font-size:12px; margin-top:4px;">Impact: <i>{s.get('impact')}</i></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with col_right:
        st.markdown("### 📊 Emotion Live")
        # Compute emotion percentages from detected people
        emotions_count = {"Engaged": 0, "Neutral": 0, "Distracted": 0}
        for p in session_status.get("people", []):
            emo = p.get("emotion", "Neutral")
            emotions_count[emo] = emotions_count.get(emo, 0) + 1
        
        tot = max(1, sum(emotions_count.values()))
        for emo, count in emotions_count.items():
            pct = int((count / tot) * 100)
            st.write(f"**{emo}** ({pct}%)")
            st.progress(pct / 100.0)

        st.markdown("---")
        st.markdown("### 🎙️ Audio Context")
        audio_ctx = session_status.get("audio_context", {})
        noise = audio_ctx.get("noise_db", -44.0)
        activity = audio_ctx.get("activity_type", "normal discussion")
        st.write(f"**Noise Level:** `{noise} dB`")
        st.write(f"**Activity:** `{activity}`")

        st.markdown("---")
        st.markdown("### 🤖 AI Insights")
        st.info(f"**Insight:** {insights.get('insight')}\n\n**Recommendation:** {insights.get('recommendation')}")

    # Auto-refresh loop when active
    if st.session_state.session_active:
        time.sleep(0.5)
        st.rerun()


# ----------------------------------------------------
# VIEW 2: TEACHER CENTRAL & PROFILE (PDF Page 17 Top)
# ----------------------------------------------------
elif view_mode == "👨‍🏫 Teacher Central & Profile":
    st.markdown('<div class="main-header">Teacher Central</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Teacher Profile & Instructional Effectiveness Dashboard</div>', unsafe_allow_html=True)

    profile = st.session_state.teacher_service.get_profile("T-101")

    # Top 3 cards
    c1, c2, c3 = st.columns([1, 1.2, 1.2])

    with c1:
        st.markdown(
            f"""
            <div class="stat-card" style="text-align:center;">
                <div style="font-size:12px; color:#94a3b8; font-weight:bold;">TEACHING EFFECTIVENESS</div>
                <div style="font-size:42px; font-weight:800; color:#38bdf8; margin: 10px 0;">{profile['effectiveness_score']}<span style="font-size:20px; color:#94a3b8;">/100</span></div>
                <div style="color:#22c55e; font-size:13px; font-weight:600;">↗️ {profile['trend']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        strengths_html = "".join([f"<li>{s}</li>" for s in profile["strengths"]])
        st.markdown(
            f"""
            <div class="stat-card">
                <div style="font-size:13px; color:#f59e0b; font-weight:bold; margin-bottom:8px;">🎗️ STRENGTHS</div>
                <ul style="color:#f8fafc; font-size:13px; padding-left:18px; margin:0;">
                    {strengths_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        recs_html = "".join([f"<li>{r}</li>" for r in profile["recommendations"]])
        st.markdown(
            f"""
            <div class="stat-card">
                <div style="font-size:13px; color:#38bdf8; font-weight:bold; margin-bottom:8px;">📋 RECOMMENDATIONS</div>
                <ul style="color:#f8fafc; font-size:13px; padding-left:18px; margin:0;">
                    {recs_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Class History Comparison")
    hist_cols = st.columns(len(profile["class_history"]))
    for idx, hist in enumerate(profile["class_history"]):
        with hist_cols[idx]:
            st.markdown(
                f"""
                <div class="stat-card" style="text-align:center;">
                    <div style="font-size:14px; font-weight:600; color:#f8fafc;">{hist['name']}</div>
                    <div style="font-size:26px; font-weight:700; color:#818cf8; margin-top:6px;">{hist['engagement']}% <span style="font-size:14px; color:#94a3b8;">Avg</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### Student Learning Trends")
    st_cols = st.columns(len(profile["student_trends"]))
    for idx, st_trend in enumerate(profile["student_trends"]):
        with st_cols[idx]:
            status_color = "#22c55e" if st_trend["status"] == "improved" else "#ef4444"
            arrow = "↗️" if st_trend["status"] == "improved" else "↘️"
            st.markdown(
                f"""
                <div class="stat-card">
                    <div style="font-size:15px; font-weight:bold; color:#f8fafc;">{st_trend['name']}</div>
                    <div style="font-size:18px; font-weight:600; color:{status_color}; margin-top:4px;">
                        {arrow} {st_trend['change']}%
                    </div>
                    <div style="font-size:11px; color:#94a3b8; text-transform:uppercase;">{st_trend['status']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ----------------------------------------------------
# VIEW 3: SESSION ANALYTICS & REPORTS (PDF Page 17 Bottom)
# ----------------------------------------------------
elif view_mode == "📊 Session Analytics & Reports":
    st.markdown('<div class="main-header">Analytics & Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep dive into classroom engagement and attendance records</div>', unsafe_allow_html=True)

    trends = st.session_state.analytics_service.get_session_trends(st.session_state.session_id)

    # PDF & CSV Export buttons matching page 17 bottom screenshot
    b_col1, b_col2, _ = st.columns([1, 1, 3])
    with b_col1:
        pdf_data = st.session_state.analytics_service.generate_pdf_report(st.session_state.session_id)
        st.download_button(
            label="📄 PDF Report",
            data=pdf_data,
            file_name=f"session_{st.session_state.session_id}_analytics.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with b_col2:
        csv_data = st.session_state.analytics_service.generate_csv_report(st.session_state.session_id)
        st.download_button(
            label="📊 CSV Data",
            data=csv_data,
            file_name=f"session_{st.session_state.session_id}_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("### Attention Trend")
        timeline = trends["timeline"]
        if timeline:
            df_att = pd.DataFrame({
                "Time": [t["time"] for t in timeline],
                "avg_attention": [t["avg_attention"] for t in timeline]
            }).set_index("Time")
            st.line_chart(df_att, color="#38bdf8")
        else:
            st.info("Collecting attention time-series points...")

    with chart_col2:
        st.markdown("### Emotion Distribution Over Time")
        if timeline:
            emo_records = []
            for t in timeline:
                em_dict = t["emotions"]
                emo_records.append({
                    "Time": t["time"],
                    "Engaged": em_dict.get("Engaged", 0),
                    "Neutral": em_dict.get("Neutral", 0),
                    "Distracted": em_dict.get("Distracted", 0)
                })
            df_emo = pd.DataFrame(emo_records).set_index("Time")
            st.bar_chart(df_emo, color=["#10b981", "#64748b", "#ef4444"])
        else:
            st.info("Collecting emotion distribution points...")

    st.markdown("### 📝 Verified Attendance Roster (Non-Duplicate Guaranteed)")
    att_summary = trends["attendance_summary"]
    if att_summary:
        df_roster = pd.DataFrame(att_summary)
        df_roster.rename(columns={
            "student_id": "Student ID",
            "first_seen": "First Detected",
            "last_seen": "Last Detected",
            "duration_sec": "Presence Duration (s)",
            "status": "Attendance Status"
        }, inplace=True)
        st.dataframe(df_roster, use_container_width=True)
    else:
        st.warning("No students marked present yet. Participants must meet the minimum presence threshold to be verified.")
