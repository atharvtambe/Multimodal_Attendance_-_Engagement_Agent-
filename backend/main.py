import os
import cv2
import json
import time
import base64
import asyncio
from datetime import datetime
from typing import Optional
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from backend.core.database import init_db, SessionLocal, SessionModel, Attendance, Student
from backend.core.face_detection import FaceDetector
from backend.core.camera_service import CameraService
from backend.core.face_recognition import FaceRecognizer
from backend.core.audio_analysis import AudioAnalyzer
from backend.core.multimodal_agent import MultiModalAgent
from backend.core.recommendations_engine import RecommendationsEngine
from backend.core.ai_suggestions import AISuggestionEngine
from backend.core.llm_insights import InsightGenerator
from backend.core.teacher_profiles import TeacherProfileService
from backend.core.analytics_service import AnalyticsService

# Initialize Database Schema
init_db()

app = FastAPI(
    title="Multimodal Attendance & Engagement Agent API",
    version="1.0.0",
    description="Multimodal system integrating computer vision and audio analysis for classroom intelligence."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global core service instances
face_detector = FaceDetector()
face_recognizer = FaceRecognizer()
camera_service = CameraService(camera_id=0)
audio_analyzer = AudioAnalyzer(sample_rate=16000)
multimodal_agent = MultiModalAgent(attendance_threshold_seconds=10.0)
recs_engine = RecommendationsEngine()
suggestion_engine = AISuggestionEngine()
insight_gen = InsightGenerator()
teacher_service = TeacherProfileService()
analytics_service = AnalyticsService()

# Current active session ID
current_session_id: Optional[int] = None

@app.on_event("startup")
def startup_event():
    global current_session_id
    db = SessionLocal()
    try:
        active_sess = db.query(SessionModel).filter(SessionModel.is_active == True).first()
        if not active_sess:
            active_sess = SessionModel(
                title="CS101 - Deep Learning & Multimodal AI",
                teacher_id="T-101",
                start_time=datetime.utcnow(),
                is_active=True
            )
            db.add(active_sess)
            db.commit()
            db.refresh(active_sess)
        current_session_id = active_sess.id
    finally:
        db.close()
    camera_service.start()


@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "active_session": current_session_id}


@app.post("/api/session/start")
def start_session(title: str = "Classroom Intelligence Session", teacher_id: str = "T-101"):
    global current_session_id
    db = SessionLocal()
    try:
        # End any previous active sessions to prevent duplicate active state
        db.query(SessionModel).filter(SessionModel.is_active == True).update({"is_active": False, "end_time": datetime.utcnow()})
        db.commit()

        new_sess = SessionModel(
            title=title,
            teacher_id=teacher_id,
            start_time=datetime.utcnow(),
            is_active=True
        )
        db.add(new_sess)
        db.commit()
        db.refresh(new_sess)
        current_session_id = new_sess.id
        return {"session_id": new_sess.id, "title": new_sess.title, "status": "started"}
    finally:
        db.close()


@app.post("/api/session/stop")
def stop_session():
    global current_session_id
    db = SessionLocal()
    try:
        if current_session_id:
            sess = db.query(SessionModel).filter(SessionModel.id == current_session_id).first()
            if sess:
                sess.is_active = False
                sess.end_time = datetime.utcnow()
                db.commit()
        ret_id = current_session_id
        current_session_id = None
        return {"session_id": ret_id, "status": "stopped"}
    finally:
        db.close()


@app.get("/api/session/{session_id}/status")
def get_session_status(session_id: int):
    frame = camera_service.get_frame()
    if frame is not None:
        detections = face_detector.process_frame(frame)
        session_status = multimodal_agent.process_session_frame(detections, face_detector, frame, session_id=session_id)
    else:
        session_status = {
            "timestamp": datetime.utcnow().timestamp(),
            "people": [],
            "active_count": 0,
            "present_count": 0,
            "avg_attention": 75.0,
            "avg_engagement": 75.0,
            "audio_context": multimodal_agent.last_audio_metric
        }

    recommendations = recs_engine.generate_realtime_recommendations(session_status)
    suggestions = suggestion_engine.generate_suggestions(session_status)
    ai_insights = insight_gen.generate_classroom_insight(session_status)

    return {
        "session_id": session_id,
        "status": session_status,
        "recommendations": recommendations,
        "suggestions": suggestions,
        "ai_insights": ai_insights
    }


@app.get("/api/session/{session_id}/trends")
def get_session_trends(session_id: int):
    return analytics_service.get_session_trends(session_id)


@app.get("/api/session/{session_id}/export/csv")
def export_csv(session_id: int):
    csv_text = analytics_service.generate_csv_report(session_id)
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=session_{session_id}_report.csv"}
    )


@app.get("/api/session/{session_id}/export/pdf")
def export_pdf(session_id: int):
    pdf_bytes = analytics_service.generate_pdf_report(session_id)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=session_{session_id}_report.pdf"}
    )


@app.get("/api/teacher/{teacher_id}/profile")
def get_teacher_profile(teacher_id: str):
    return teacher_service.get_profile(teacher_id)


# WebSocket for Audio streaming (Activity 3.1 & Page 10 screenshot)
@app.websocket("/ws/audio")
async def audio_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            metrics = audio_analyzer.analyze_audio_chunk(data)
            if metrics:
                multimodal_agent.update_audio_state(metrics)
                await websocket.send_json(metrics)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Audio WS error: {e}")


# MJPEG Video Feed Stream for live browser / dashboard preview
def generate_live_stream_frames():
    while True:
        frame = camera_service.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        detections = face_detector.process_frame(frame)
        status = multimodal_agent.process_session_frame(detections, face_detector, frame, session_id=current_session_id)

        # Draw bounding boxes and status labels matching page 16 screenshot: "ID: 1 | engaged"
        for p in status.get("people", []):
            x, y, w, h = p["bbox"]
            is_engaged = p.get("status") == "engaged"
            color = (50, 205, 50) if is_engaged else (30, 105, 255) # Green vs Orange-Red
            label = f"ID: {p['id']} | {p['status']}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Label background banner
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, max(0, y - th - 10)), (x + tw + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_live_stream_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
