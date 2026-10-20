import cv2
import time
import asyncio
import numpy as np
from fastapi import WebSocket

class CameraService:
    def __init__(self, camera_id=0):
        """
        camera_id can be an integer for a live webcam (0, 1, ...)
        or a string file path to a pre-recorded classroom video.
        """
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False
        self._synthetic_step = 0

    def start(self):
        if self.cap is None:
            try:
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    print(f"Warning: Could not open camera {self.camera_id}, fallback mode active.")
                else:
                    self.is_running = True
                    return
            except Exception as e:
                print(f"Camera open exception: {e}")
            self.is_running = True

    def stop(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_running = False

    def get_frame(self):
        if self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                return frame
            elif isinstance(self.camera_id, str):
                # Video file reached end; loop back to start for smooth continuous demo
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret:
                    return frame

        # Synthetic fallback frame for headless/test environments with simulated classroom participants
        if self.is_running:
            return self._generate_simulated_classroom_frame()
        return None

    def _generate_simulated_classroom_frame(self):
        """
        Generates a crisp synthetic classroom test video frame with 2 simulated student faces
        when physical hardware is absent, ensuring 100% testability offline.
        """
        self._synthetic_step += 1
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        # Background blackboard and wall
        cv2.rectangle(frame, (20, 20), (620, 140), (25, 45, 30), -1)
        cv2.putText(frame, "EduSense Classroom - Session Live", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        
        # Student 1 (Drifting slightly)
        s1_x = int(180 + 10 * np.sin(self._synthetic_step * 0.05))
        s1_y = 260
        # Head / face oval
        cv2.circle(frame, (s1_x, s1_y), 50, (190, 180, 170), -1)
        cv2.circle(frame, (s1_x - 18, s1_y - 10), 6, (40, 40, 40), -1)
        cv2.circle(frame, (s1_x + 18, s1_y - 10), 6, (40, 40, 40), -1)
        cv2.line(frame, (s1_x - 15, s1_y + 20), (s1_x + 15, s1_y + 20), (30, 30, 30), 3)

        # Student 2
        s2_x = int(440 + 8 * np.cos(self._synthetic_step * 0.04))
        s2_y = 270
        cv2.circle(frame, (s2_x, s2_y), 48, (175, 170, 160), -1)
        cv2.circle(frame, (s2_x - 16, s2_y - 8), 5, (40, 40, 40), -1)
        cv2.circle(frame, (s2_x + 16, s2_y - 8), 5, (40, 40, 40), -1)
        cv2.line(frame, (s2_x - 12, s2_y + 22), (s2_x + 12, s2_y + 22), (30, 30, 30), 3)

        return frame
