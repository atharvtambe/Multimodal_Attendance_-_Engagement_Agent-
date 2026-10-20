import argparse
import subprocess
import sys
import os

def run_backend(port=8000):
    print(f"[EduSense] Starting Multimodal Backend on http://localhost:{port} ...")
    cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", str(port), "--reload"]
    subprocess.run(cmd)

def run_dashboard(port=8501):
    print(f"[EduSense] Starting Streamlit Dashboard on http://localhost:{port} ...")
    cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", str(port)]
    subprocess.run(cmd)

def run_tests():
    print("[EduSense] Running Multimodal Test Suite...")
    cmd = [sys.executable, "-m", "unittest", "tests/test_pipeline.py"]
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Multimodal Attendance & Engagement Agent Runner")
    parser.add_argument("mode", choices=["dashboard", "backend", "test", "all"], default="dashboard", nargs="?",
                        help="Mode to run: 'dashboard' (Streamlit UI), 'backend' (FastAPI REST & WS), 'test' (Unit tests), or 'all'")
    args = parser.parse_args()

    if args.mode == "backend":
        run_backend()
    elif args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "test":
        run_tests()
    elif args.mode == "all":
        # Launch backend and dashboard together
        import threading
        t_back = threading.Thread(target=run_backend, daemon=True)
        t_back.start()
        run_dashboard()

if __name__ == "__main__":
    main()
