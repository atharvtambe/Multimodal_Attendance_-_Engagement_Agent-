import os
import unittest
import numpy as np
from datetime import datetime

from backend.core.database import init_db, SessionLocal, SessionModel, Attendance, SessionPerson, PersonMetric, Student
from backend.core.face_detection import FaceDetector
from backend.core.camera_service import CameraService
from backend.core.face_recognition import FaceRecognizer
from backend.core.audio_analysis import AudioAnalyzer
from backend.core.multimodal_agent import MultiModalAgent
from backend.core.recommendations_engine import RecommendationsEngine
from backend.core.teacher_profiles import TeacherProfileService
from backend.core.analytics_service import AnalyticsService

class TestMultimodalAttendancePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self.db = SessionLocal()
        # Create fresh test session
        self.session = SessionModel(title="Unit Test Session", is_active=True)
        self.db.add(self.session)
        self.db.commit()
        self.db.refresh(self.session)

    def tearDown(self):
        self.db.query(PersonMetric).filter(PersonMetric.session_id == self.session.id).delete()
        self.db.query(Attendance).filter(Attendance.session_id == self.session.id).delete()
        self.db.query(SessionPerson).filter(SessionPerson.session_id == self.session.id).delete()
        self.db.query(SessionModel).filter(SessionModel.id == self.session.id).delete()
        self.db.commit()
        self.db.close()

    def test_milestone2_vision_and_camera(self):
        detector = FaceDetector()
        cam = CameraService()
        cam.start()
        frame = cam.get_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        cam.stop()

        detections = detector.process_frame(frame)
        self.assertIsInstance(detections, list)
        att, emo = detector.estimate_head_pose_and_attention(frame, (100, 100, 80, 80))
        self.assertGreaterEqual(att, 0.0)
        self.assertLessEqual(att, 100.0)
        self.assertIn(emo, ["Engaged", "Neutral", "Distracted"])

    def test_milestone3_audio_analysis(self):
        analyzer = AudioAnalyzer(sample_rate=16000)
        # Synthetic sine wave simulating speech chunk
        t = np.linspace(0, 0.5, 8000, endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        tone_bytes = (tone * 32767).astype(np.int16).tobytes()

        metrics = analyzer.analyze_audio_chunk(tone_bytes)
        self.assertIsNotNone(metrics)
        self.assertIn("noise_db", metrics)
        self.assertIn("speech_ratio", metrics)
        self.assertIn("activity_type", metrics)
        self.assertGreater(metrics["noise_db"], -60)

    def test_milestone4_multimodal_fusion_and_deduplication(self):
        # Strict test: Verify no duplicate data is created for the same participant
        agent = MultiModalAgent(attendance_threshold_seconds=1.0)
        detector = FaceDetector()
        cam = CameraService()
        cam.start()
        frame = cam.get_frame()

        synthetic_detections = [{"bbox": (180, 240, 60, 60), "score": 1.0}]

        # Run multiple consecutive frames (simulating presence over time)
        for _ in range(5):
            res = agent.process_session_frame(
                detections=synthetic_detections,
                detector_obj=detector,
                raw_frame=frame,
                session_id=self.session.id
            )

        cam.stop()

        # Check in DB that attendance record is strictly UNIQUE (NO DUPLICATES!)
        attendances = self.db.query(Attendance).filter(Attendance.session_id == self.session.id).all()
        # Even after multiple frames, there should be exactly ONE attendance record for this student
        student_ids = [a.student_id for a in attendances]
        self.assertEqual(len(student_ids), len(set(student_ids)), "Duplicate attendance records detected!")

        session_persons = self.db.query(SessionPerson).filter(SessionPerson.session_id == self.session.id).all()
        track_ids = [sp.track_id for sp in session_persons]
        self.assertEqual(len(track_ids), len(set(track_ids)), "Duplicate session person tracks detected!")

    def test_milestone5_analytics_and_reports(self):
        analytics = AnalyticsService()
        trends = analytics.get_session_trends(self.session.id)
        self.assertIn("timeline", trends)
        self.assertIn("attendance_summary", trends)

        csv_report = analytics.generate_csv_report(self.session.id)
        self.assertIn("SESSION ATTENDANCE SUMMARY", csv_report)

        pdf_bytes = analytics.generate_pdf_report(self.session.id)
        self.assertGreater(len(pdf_bytes), 100)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_milestone5_recommendations_and_teacher_profile(self):
        recs_engine = RecommendationsEngine()
        status = {
            "avg_attention": 35.0,
            "avg_engagement": 40.0,
            "people": [{"id": 1, "attention": 30}],
            "audio_context": {"noise_db": -25.0, "activity_type": "noisy"}
        }
        recs = recs_engine.generate_realtime_recommendations(status)
        self.assertTrue(len(recs) > 0)

        teacher_svc = TeacherProfileService()
        profile = teacher_svc.get_profile("T-101")
        self.assertEqual(profile["teacher_id"], "T-101")
        self.assertIn("effectiveness_score", profile)
        self.assertIn("strengths", profile)
        self.assertIn("class_history", profile)

if __name__ == "__main__":
    unittest.main()
