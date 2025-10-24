# src/main.py
import argparse
import uvicorn

SERVICES = {
    "nlp_simple": "nlp.app:app",
    "asr": "python_services.asr.app:app",
    "diar": "python_services.diarization.app:app",
    "emotion": "python_services.emotion.app:app",
    "nlp": "python_services.nlp.app:app",
    "nonverb": "python_services.nonverb_features.app:app",
    "viz": "python_services.visualization.app:app",
    "robot": "python_services.robot_data.app:app",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("service", choices=SERVICES.keys())
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(SERVICES[args.service], host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
