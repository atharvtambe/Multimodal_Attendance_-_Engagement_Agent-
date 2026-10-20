# Multimodal Attendance & Engagement Agent (EduSense AI)

A multimodal AI classroom assistant integrating **Computer Vision**, **Audio Signal Processing**, and **Rule-Based Agent Reasoning** to provide reliable presence-based attendance tracking and deep, interpretable student engagement analytics.

Built strictly according to project specifications and milestones outlined in the project documentation.

---

## 📌 Project Architecture & PDF Milestones Alignment

| Milestone | Activity & Component | Implementation |
|---|---|---|
| **Milestone 1: Requirements Specification & Setup** | Activity 1.1 & 1.2: Environment & Dependencies | [`backend/requirements.txt`](file:///c:/Users/ATHARV/OneDrive/Documents/Desktop/python_in_vscode/Multimodal_Attendance_&_Engagement_Agent/backend/requirements.txt), SQLite Database Schema |
| **Milestone 2: Vision Module Development** | Activity 2.1: Face detection & tracking<br>Activity 2.2: Presence-based attendance logic | [`backend/core/face_detection.py`](file:///c:/Users/ATHARV/OneDrive/Documents/Desktop/python_in_vscode/Multimodal_Attendance_&_Engagement_Agent/backend/core/face_detection.py)<br>[`backend/core/camera_service.py`](file:///c:/Users/ATHARV/OneDrive/Documents/Desktop/python_in_vscode/Multimodal_Attendance_&_Engagement_Agent/backend/core/camera_service.py)<br>[`backend/core/face_recognition.py`](file:///c:/Users/ATHARV/OneDrive/Documents/Desktop/python_in_vscode/Multimodal_Attendance_&_Engagement_Agent/backend/core/face_recognition.py) |
| **Milestone 3: Audio Module Development** | Activity 3.1: Audio signal extraction<br>Activity 3.2: Noise & speech activity analysis | [`backend/core/audio_analysis.py`](file:///c:/Users/ATHARV/OneDrive/Documents/Desktop/python_in_vscode/Multimodal_Attendance_&_Engagement_Agent/backend/core/audio_analysis.py)<br>[`frontend/src/components/AudioCapture.jsx`](file:///c:/Users/ATHARV/OneDrive/Documents/Desktop/python_in_vscode/Multimodal_Attendance_&_Engagement_Agent/frontend/src/components/AudioCapture.jsx) |
| **Milestone 4: Multimodal Agent Implementation** | Activity 4.1: Multimodal signal fusion<br>Activity 4.2: Engagement scoring logic | [`backend/core/multimodal_agent.py`](file:///c:/Users/ATHARV/OneDrive/Documents/Desktop/python_in_vscode/Multimodal_Attendance_&_Engagement_Agent/backend/core/multimodal_agent.py)<br>Temporal synchronization of vision & acoustic signals |
| **Milestone 5: Analytics & Recommendation Engine** | Activity 5.1: Teaching suggestion logic<br>Activity 5.2: Student analytics generation | [`backend/core/recommendations_engine.py`](file:///c:/Users/ATHARV/OneDrive/Documents/Desktop/python_in_vscode/Multimodal_Attendance_&_Engagement_Agent/backend/core/recommendations_engine.py)<br>[`backend/core/ai_suggestions.py`](file:///c:/Users/ATHARV/OneDrive/Documents/Desktop/python_in_vscode/Multimodal_Attendance_&_Engagement_Agent/backend/core/ai_suggestions.py)<br>[`backend/core/llm_insights.py`](file:///c:/Users/ATHARV/OneDrive/Documents/Desktop/python_in_vscode/Multimodal_Attendance_&_Engagement_Agent/backend/core/llm_insights.py)<br>[`backend/core/teacher_profiles.py`](file:///c:/Users/ATHARV/OneDrive/Documents/Desktop/python_in_vscode/Multimodal_Attendance_&_Engagement_Agent/backend/core/teacher_profiles.py)<br>[`backend/core/analytics_service.py`](file:///c:/Users/ATHARV/OneDrive/Documents/Desktop/python_in_vscode/Multimodal_Attendance_&_Engagement_Agent/backend/core/analytics_service.py) |
| **Milestone 6: Interface, Testing & Deployment** | Activity 6.1: Run core backend<br>Activity 6.2: Launch Streamlit dashboard<br>Activity 6.3: Validate edge cases | [`backend/main.py`](file:///c:/Users/ATHARV/OneDrive/Documents/Desktop/python_in_vscode/Multimodal_Attendance_&_Engagement_Agent/backend/main.py)<br>[`streamlit_app.py`](file:///c:/Users/ATHARV/OneDrive/Documents/Desktop/python_in_vscode/Multimodal_Attendance_&_Engagement_Agent/streamlit_app.py)<br>[`tests/test_pipeline.py`](file:///c:/Users/ATHARV/OneDrive/Documents/Desktop/python_in_vscode/Multimodal_Attendance_&_Engagement_Agent/tests/test_pipeline.py) |

---

## 🔒 Strict Deduplication Guarantee ("No Duplicate Data")

As per requirements, duplicate data is strictly prohibited at both the schema and runtime levels:
1. **Attendance Deduplication**: The `attendances` table enforces a unique constraint `UNIQUE(session_id, student_id)`. Continuous presence increments cumulative duration without creating redundant rows.
2. **Participant Tracking Deduplication**: The `session_persons` table enforces `UNIQUE(session_id, track_id)`. Face detections across consecutive video frames are continuously associated via spatial IoU tracking.
3. **Database Seeding Idempotency**: Teacher and student registration checks exist before insertion; re-initialization does not create duplicate entries.
4. **Export Cleanliness**: All analytics exports (CSV and PDF) generate deduplicated participant rosters and downsampled timeline metrics.

---

## 🚀 Quickstart & Running the Application

### 1. Run All Tests
```bash
python run.py test
```
*Runs the 5 milestone test suites verifying vision, audio, fusion, deduplication, and PDF/CSV reporting.*

### 2. Launch the Interactive Streamlit Dashboard
```bash
python run.py dashboard
```
*Or directly: `streamlit run streamlit_app.py`*
*Open http://localhost:8501 in your browser.*

### 3. Launch the FastAPI Backend & Streaming Server
```bash
python run.py backend
```
*Or directly: `uvicorn backend.main:app --reload --port 8000`*
- Interactive Swagger UI: `http://localhost:8000/docs`
- Video MJPEG Stream: `http://localhost:8000/video_feed`
- Real-time Audio WebSocket: `ws://localhost:8000/ws/audio`

---

## 🖥️ User Interface Overview (Matching PDF Screenshots)

1. **Live Classroom Intelligence** (PDF Page 16):
   - **Real-time Video**: Live webcam or simulated classroom with bounding boxes (`ID: 1 | engaged`).
   - **Status Banner**: Real-time optimal engagement alerts (`Engagement is optimal. Keep going!`).
   - **Live Leaderboard**: Student engagement points.
   - **AI Suggestion Box**: Context-aware high-priority instructional cues (`Challenge students with harder problem - Impact: Deep Learning`).
   - **Emotion Live**: Distribution of Engaged, Neutral, and Distracted states.
   - **Audio Context**: Ambient noise in decibels (`-44.0 dB`) and activity classification (`normal discussion`).
   - **AI Insights**: Automated diagnostic observations and pedagogical suggestions.

2. **Teacher Central & Profile** (PDF Page 17 Top):
   - **Teaching Effectiveness Score**: `87.5 / 100` with `+12%` trend.
   - **Strengths & Recommendations**: Identified strengths and actionable strategies.
   - **Class History Comparison**: Multi-class engagement averages.
   - **Student Learning Trends**: Individual progress tracking (`Priya +15%`, `Kumar +8%`, `Asha -5%`).

3. **Session Analytics & Reports** (PDF Page 17 Bottom):
   - **Attention Trend**: Interactive temporal line chart.
   - **Emotion Distribution Over Time**: Stacked/grouped bar charts.
   - **Verified Attendance Roster**: Clean, deduplicated table of present students with arrival timestamps and durations.
   - **One-Click Exports**: Generate official **PDF Reports** and download **CSV Data**.
