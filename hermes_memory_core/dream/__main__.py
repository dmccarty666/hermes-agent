#!/usr/bin/env python3
"""Entry point for ``python -m hermes_memory_core.dream``."""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from hermes_memory_core.dream.worker import DreamWorker

# Dedicated log file per Phase 1 conventions — separate from memory-*.log
_LOG_PATH = Path.home() / ".hermes" / "logs" / "memory-dream.log"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hermes Local Memory — Dreamer")
    parser.add_argument(
        "--scope",
        default="since_last",
        choices=["session", "project", "since_last", "all"],
        help="Which sessions to process (default: since_last)",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session ID (required when scope=session)",
    )
    parser.add_argument(
        "--llm-endpoint",
        default="http://192.168.2.105:1234",
        help="LMS/Spark2 inference server URL",
    )
    parser.add_argument(
        "--llm-model",
        default="Qwen3.6-35B",
        help="Model identifier",
    )
    args = parser.parse_args()

    # File handler for dedicated dream log
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(_LOG_PATH),
            logging.StreamHandler(sys.stderr),
        ],
    )
    logger = logging.getLogger("hermes_memory_core.dream")
    logger.info("Dreamer starting — scope=%s", args.scope)

    worker = DreamWorker(
        llm_endpoint=args.llm_endpoint,
        llm_model=args.llm_model,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    try:
        result = worker.dream(scope=args.scope, session_id=args.session_id)
        # DreamWorker._stage_record_dream_run already wrote the report to
        # ~/.hermes/memory/dreams/YYYY-MM-DD-HHMM.md — don't double-write.
        report_path = result.output_path or "(no report — empty scope)"
        logger.info(
            "Dream complete — report=%s facts=%d decisions=%d questions=%d",
            report_path,
            result.dream_run.facts_created,
            result.dream_run.decisions_created,
            result.dream_run.questions_created,
        )
        sys.exit(0)
    except Exception as exc:
        logger.exception("Dreamer failed: %s", exc)
        sys.exit(1)
