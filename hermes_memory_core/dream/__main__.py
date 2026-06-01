"""Entry point for: python -m hermes_memory_core.dream --scope <scope>"""
import argparse
import json
import logging

from hermes_memory_core.dream.worker import DreamWorker

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Hermes Dream Worker")
    parser.add_argument(
        "--scope",
        default="since_last",
        choices=["since_last", "today", "date", "project", "weekly"],
        help="Scope of the dream run",
    )
    parser.add_argument("--date", help="ISO date for scope=date (YYYY-MM-DD)")
    parser.add_argument("--project", help="Project name for scope=project")
    parser.add_argument("--deep", action="store_true", help="Deep analysis mode")
    args = parser.parse_args()

    worker = DreamWorker()
    result = worker.run(
        scope=args.scope,
        deep=args.deep,
        date=args.date,
        project=args.project,
    )
    print(json.dumps(result, indent=2))
