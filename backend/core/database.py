import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, UniqueConstraint, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "attendance_agent.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def utc_now():
    return datetime.now(timezone.utc)

class Teacher(Base):
    __tablename__ = "teachers"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=utc_now)

    sessions = relationship("SessionModel", back_populates="teacher")


class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    teacher_id = Column(String, ForeignKey("teachers.id"), nullable=True)
    title = Column(String, default="Classroom Session")
    start_time = Column(DateTime, default=utc_now)
    end_time = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    avg_attention = Column(Float, default=0.0)
    avg_engagement = Column(Float, default=0.0)

    teacher = relationship("Teacher", back_populates="sessions")
    persons = relationship("SessionPerson", back_populates="session", cascade="all, delete-orphan")
    attendances = relationship("Attendance", back_populates="session", cascade="all, delete-orphan")
    metrics = relationship("PersonMetric", back_populates="session", cascade="all, delete-orphan")


class Student(Base):
    __tablename__ = "students"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=utc_now)

    attendances = relationship("Attendance", back_populates="student")


class Attendance(Base):
    __tablename__ = "attendances"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, index=True)
    student_id = Column(String, ForeignKey("students.id"), nullable=False, index=True)
    first_seen = Column(DateTime, default=utc_now)
    last_seen = Column(DateTime, default=utc_now)
    presence_duration_seconds = Column(Float, default=0.0)
    status = Column(String, default="Present") # Present, Partial, Absent

    session = relationship("SessionModel", back_populates="attendances")
    student = relationship("Student", back_populates="attendances")

    # Strictly prevent duplicate attendance entries for same session and student
    __table_args__ = (
        UniqueConstraint("session_id", "student_id", name="uq_session_student_attendance"),
    )


class SessionPerson(Base):
    __tablename__ = "session_persons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, index=True)
    track_id = Column(Integer, nullable=False, index=True)
    student_id = Column(String, nullable=True)
    first_seen = Column(DateTime, default=utc_now)
    last_seen = Column(DateTime, default=utc_now)
    total_presence_sec = Column(Float, default=0.0)
    is_marked_present = Column(Boolean, default=False)

    session = relationship("SessionModel", back_populates="persons")

    # Strictly prevent duplicate person track entry for the same session
    __table_args__ = (
        UniqueConstraint("session_id", "track_id", name="uq_session_track"),
    )


class PersonMetric(Base):
    __tablename__ = "person_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, index=True)
    track_id = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, default=utc_now, index=True)
    attention_score = Column(Float, default=0.0)
    engagement_score = Column(Float, default=0.0)
    emotion = Column(String, default="Neutral")
    movement_score = Column(Float, default=0.0)
    speech_score = Column(Float, default=0.0)

    session = relationship("SessionModel", back_populates="metrics")


class ClassroomAudioMetric(Base):
    __tablename__ = "classroom_audio_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=utc_now, index=True)
    noise_db = Column(Float, default=-40.0)
    speech_ratio = Column(Float, default=0.0)
    activity_type = Column(String, default="normal discussion")


def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Seed default teacher if not exists (Strictly non-duplicate check)
        teacher = db.query(Teacher).filter(Teacher.id == "T-101").first()
        if not teacher:
            teacher = Teacher(id="T-101", name="Prof. Sharma", email="prof.sharma@example.edu")
            db.add(teacher)

        # Seed initial students if not exist (Strictly non-duplicate check)
        initial_students = [
            ("student_1", "Student 1 (Arun)"),
            ("student_2", "Student 2 (Priya)"),
            ("student_3", "Student 3 (Kumar)"),
            ("student_4", "Student 4 (Asha)"),
        ]
        for s_id, s_name in initial_students:
            if not db.query(Student).filter(Student.id == s_id).first():
                db.add(Student(id=s_id, name=s_name))

        db.commit()
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully without duplicate data.")
