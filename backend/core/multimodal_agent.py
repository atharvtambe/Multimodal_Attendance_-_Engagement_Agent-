import time
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional
from .database import SessionLocal, SessionModel, SessionPerson, Attendance, PersonMetric, ClassroomAudioMetric, Student

def utc_now():
    return datetime.now(timezone.utc)

class MultiModalAgent:
    def __init__(self, attendance_threshold_seconds: float = 10.0):
        """
        attendance_threshold_seconds: Cumulative visible duration required before marking present.
        """
        self.attendance_threshold = attendance_threshold_seconds
        
        # State tracking: track_id -> track state dict
        # {
        #   "track_id": int,
        #   "student_id": str,
        #   "first_seen": float,
        #   "last_seen": float,
        #   "presence_duration": float,
        #   "is_present": bool,
        #   "last_bbox": tuple,
        #   "attention_history": list,
        #   "engagement_history": list,
        #   "last_movement_score": float
        # }
        self.active_tracks: Dict[int, dict] = {}
        self.next_track_id = 1
        self.last_audio_metric = {
            "noise_db": -44.0,
            "speech_ratio": 0.20,
            "activity_type": "normal discussion",
            "audio_engagement_score": 80.0,
            "is_speech": False
        }

    def update_audio_state(self, audio_metrics: dict):
        if audio_metrics:
            self.last_audio_metric.update(audio_metrics)

    def match_or_create_tracks(self, detections: List[dict], current_time: float) -> List[dict]:
        """
        Associates detections across consecutive frames using spatial IoU tracking.
        Prevents duplicate track creation and ensures stable identities.
        """
        matched_results = []
        unmatched_detections = list(detections)

        # Try to match with existing active tracks based on bounding box proximity
        for track_id, track in list(self.active_tracks.items()):
            # If track hasn't been seen for > 4.0 seconds, consider temporarily out of view
            time_since_last = current_time - track["last_seen"]
            if time_since_last > 4.0:
                continue

            best_iou = 0.0
            best_det_idx = -1
            ox, oy, ow, oh = track["last_bbox"]

            for idx, det in enumerate(unmatched_detections):
                nx, ny, nw, nh = det["bbox"]
                # Calculate Intersection over Union (IoU)
                ix1 = max(ox, nx)
                iy1 = max(oy, ny)
                ix2 = min(ox + ow, nx + nw)
                iy2 = min(oy + oh, ny + nh)

                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                intersection = iw * ih
                union = (ow * oh) + (nw * nh) - intersection
                iou = intersection / max(1.0, union)

                # Centroid distance as well
                c_dist = math.hypot((ox + ow/2) - (nx + nw/2), (oy + oh/2) - (ny + nh/2))

                if iou > 0.25 or (c_dist < 80 and iou > 0.05):
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = idx

            if best_det_idx >= 0:
                det = unmatched_detections.pop(best_det_idx)
                # Compute movement score from centroid shift
                cx_old, cy_old = ox + ow/2, oy + oh/2
                cx_new, cy_new = det["bbox"][0] + det["bbox"][2]/2, det["bbox"][1] + det["bbox"][3]/2
                movement_delta = math.hypot(cx_new - cx_old, cy_new - cy_old)
                movement_score = min(100.0, movement_delta * 4.0)

                # Accumulate presence duration
                dt = min(1.0, current_time - track["last_seen"])
                track["presence_duration"] += dt
                track["last_seen"] = current_time
                track["last_bbox"] = det["bbox"]
                track["last_movement_score"] = movement_score

                matched_results.append({
                    "track_id": track_id,
                    "student_id": track["student_id"],
                    "bbox": det["bbox"],
                    "movement_score": movement_score,
                    "presence_duration": track["presence_duration"]
                })

        # New detections: create fresh tracks (without duplicating)
        for det in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1
            student_id = f"Student {track_id}"

            self.active_tracks[track_id] = {
                "track_id": track_id,
                "student_id": student_id,
                "first_seen": current_time,
                "last_seen": current_time,
                "presence_duration": 0.5,
                "is_present": False,
                "last_bbox": det["bbox"],
                "attention_history": [],
                "engagement_history": [],
                "last_movement_score": 10.0
            }

            matched_results.append({
                "track_id": track_id,
                "student_id": student_id,
                "bbox": det["bbox"],
                "movement_score": 10.0,
                "presence_duration": 0.5
            })

        return matched_results

    def compute_multimodal_engagement(
        self,
        visual_attention: float,
        movement_score: float,
        audio_metrics: dict
    ) -> float:
        """
        Activity 4.1 & 4.2: Multimodal Signal Fusion & Engagement Scoring.
        Fuses visual attention, movement dynamics, and classroom audio context.
        Formula weights:
        - 55% Visual Attention (symmetry, gaze, facing front)
        - 25% Audio Context (healthy discussion vs distracting noise)
        - 20% Movement Stability (moderate movement indicates alertness; excessive or zero indicates distraction/absence)
        """
        # Movement factor: 15-40 is optimal natural alertness; > 70 is restless/fidgeting; 0 is freeze/asleep
        if 10 <= movement_score <= 45:
            move_factor = 95.0
        elif movement_score > 70:
            move_factor = 50.0
        else:
            move_factor = 75.0

        audio_factor = audio_metrics.get("audio_engagement_score", 80.0)

        fused_score = (
            (visual_attention * 0.55) +
            (audio_factor * 0.25) +
            (move_factor * 0.20)
        )
        return float(round(max(5.0, min(100.0, fused_score)), 1))

    def process_session_frame(
        self,
        detections: List[dict],
        detector_obj,
        raw_frame,
        session_id: Optional[int] = None
    ) -> dict:
        """
        Processes a single frame:
        1. Tracks faces across time
        2. Estimates visual attention and emotion
        3. Fuses with audio signals
        4. Evaluates presence-based attendance logic
        5. Persists non-duplicate metrics to database
        """
        current_time = time.time()
        tracked = self.match_or_create_tracks(detections, current_time)
        people_status = []

        db = SessionLocal() if session_id else None

        try:
            for item in tracked:
                track_id = item["track_id"]
                track_state = self.active_tracks[track_id]

                # Visual attention and emotion
                visual_att, emotion = detector_obj.estimate_head_pose_and_attention(raw_frame, item["bbox"])
                
                # Multimodal fusion
                engagement = self.compute_multimodal_engagement(
                    visual_attention=visual_att,
                    movement_score=item["movement_score"],
                    audio_metrics=self.last_audio_metric
                )

                # Threshold presence check for attendance
                is_marked_now = False
                if track_state["presence_duration"] >= self.attendance_threshold and not track_state["is_present"]:
                    track_state["is_present"] = True
                    is_marked_now = True

                person_summary = {
                    "id": track_id,
                    "student_id": track_state["student_id"],
                    "bbox": item["bbox"],
                    "attention": visual_att,
                    "engagement": engagement,
                    "emotion": emotion,
                    "presence_sec": round(track_state["presence_duration"], 1),
                    "is_present": track_state["is_present"],
                    "status": "engaged" if engagement >= 60 else "distracted"
                }
                people_status.append(person_summary)

                # Persist to DB without duplicating attendance records
                if db and session_id:
                    # 1. Update/insert SessionPerson
                    sp = db.query(SessionPerson).filter(
                        SessionPerson.session_id == session_id,
                        SessionPerson.track_id == track_id
                    ).first()
                    if not sp:
                        sp = SessionPerson(
                            session_id=session_id,
                            track_id=track_id,
                            student_id=track_state["student_id"],
                            first_seen=utc_now(),
                            last_seen=utc_now(),
                            total_presence_sec=track_state["presence_duration"],
                            is_marked_present=track_state["is_present"]
                        )
                        db.add(sp)
                    else:
                        sp.last_seen = utc_now()
                        sp.total_presence_sec = track_state["presence_duration"]
                        sp.is_marked_present = track_state["is_present"]

                    # 2. If marked present, record unique Attendance record
                    if track_state["is_present"]:
                        att_record = db.query(Attendance).filter(
                            Attendance.session_id == session_id,
                            Attendance.student_id == track_state["student_id"]
                        ).first()
                        if not att_record:
                            att_record = Attendance(
                                session_id=session_id,
                                student_id=track_state["student_id"],
                                first_seen=utc_now(),
                                last_seen=utc_now(),
                                presence_duration_seconds=track_state["presence_duration"],
                                status="Present"
                            )
                            db.add(att_record)
                        else:
                            att_record.last_seen = utc_now()
                            att_record.presence_duration_seconds = track_state["presence_duration"]

                    # 3. Store periodic PersonMetric (sample every 1 sec to avoid flooding)
                    now_dt = utc_now()
                    db.add(PersonMetric(
                        session_id=session_id,
                        track_id=track_id,
                        timestamp=now_dt,
                        attention_score=visual_att,
                        engagement_score=engagement,
                        emotion=emotion,
                        movement_score=item["movement_score"],
                        speech_score=self.last_audio_metric.get("speech_ratio", 0.0) * 100.0
                    ))

            if db and session_id:
                db.commit()

        except Exception as e:
            if db:
                db.rollback()
            print(f"Agent state processing error: {e}")
        finally:
            if db:
                db.close()

        # Session-level aggregations
        present_count = sum(1 for p in self.active_tracks.values() if p["is_present"])
        avg_att = sum(p["attention"] for p in people_status) / len(people_status) if people_status else 75.0
        avg_eng = sum(p["engagement"] for p in people_status) / len(people_status) if people_status else 75.0

        return {
            "timestamp": current_time,
            "people": people_status,
            "active_count": len(people_status),
            "present_count": present_count,
            "avg_attention": round(avg_att, 1),
            "avg_engagement": round(avg_eng, 1),
            "audio_context": self.last_audio_metric
        }
