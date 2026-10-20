import os
import shutil
import cv2
import numpy as np

class FaceRecognizer:
    def __init__(self, db_path="student_db"):
        self.db_path = db_path
        if not os.path.exists(db_path):
            os.makedirs(db_path, exist_ok=True)
        self.has_deepface = False
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            self.has_deepface = True
        except Exception:
            self.DeepFace = None
            self.has_deepface = False

        # In-memory feature cache to prevent duplicate computation and support lightweight recognition
        self.student_templates = {}
        self._load_local_templates()

    def _load_local_templates(self):
        """Loads registered student face images into memory."""
        if not os.path.exists(self.db_path):
            return
        for file in os.listdir(self.db_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                student_id = os.path.splitext(file)[0]
                img_path = os.path.join(self.db_path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize to standard thumbnail 100x100 grayscale
                    gray = cv2.cvtColor(cv2.resize(img, (100, 100)), cv2.COLOR_BGR2GRAY)
                    self.student_templates[student_id] = gray

    def register_student(self, img_source, student_id: str):
        """
        Registers a student face without duplicating entries.
        img_source can be a file path or a numpy image array.
        """
        dest_filename = f"{student_id}.jpg"
        dest_path = os.path.join(self.db_path, dest_filename)

        if isinstance(img_source, str) and os.path.exists(img_source):
            shutil.copyfile(img_source, dest_path)
            img = cv2.imread(dest_path)
        elif isinstance(img_source, np.ndarray):
            cv2.imwrite(dest_path, img_source)
            img = img_source
        else:
            return False

        if img is not None:
            gray = cv2.cvtColor(cv2.resize(img, (100, 100)), cv2.COLOR_BGR2GRAY)
            self.student_templates[student_id] = gray
            return True
        return False

    def recognize(self, frame):
        """
        Recognizes faces present in the frame against registered database.
        Returns list of recognized student IDs with match confidence.
        """
        try:
            if not os.path.exists(self.db_path) or not os.listdir(self.db_path):
                return []

            if self.has_deepface and self.DeepFace:
                try:
                    dfs = self.DeepFace.find(img_path=frame, db_path=self.db_path, silent=True, enforce_detection=False)
                    matches = []
                    for df in dfs:
                        if not df.empty and 'identity' in df.columns:
                            identity_path = df.iloc[0]['identity']
                            student_id = os.path.splitext(os.path.basename(identity_path))[0]
                            dist = df.iloc[0].get('distance', 0.3)
                            matches.append({"student_id": student_id, "confidence": max(0.5, 1.0 - dist)})
                    if matches:
                        return matches
                except Exception:
                    pass

            # Fast lightweight template/histogram recognition fallback
            if not self.student_templates:
                return []

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            matches = []
            for student_id, tmpl in self.student_templates.items():
                res = cv2.matchTemplate(gray_frame, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > 0.45:
                    matches.append({"student_id": student_id, "confidence": round(float(max_val), 2)})

            return matches
        except Exception as e:
            print(f"Recognition error: {e}")
            return []
