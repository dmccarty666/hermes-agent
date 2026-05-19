# Copyright 2026 David McCarty. All rights reserved.
"""Project memory file generator — stage 6 of dreamer pipeline.

Updates project memory markdown files with new facts, decisions, and questions
extracted from a dream run.

Preserves manual content above `<!-- AUTO-GENERATED BELOW -->` marker.
Only modifies content below the marker.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Memory project files to update
_PROJECT_MEMORY_FILES = [
    "memory.md",
    "facts.md",
    "decisions.md",
    "open_questions.md",
    "timeline.md",
]

# Marker that separates manual content from auto-generated content
_AUTO_GENERATED_MARKER = "<!-- AUTO-GENERATED BELOW -->"

# Path to the prompts directory
_PROMPTS_DIR = Path.home() / ".hermes" / "memory" / "prompts"

# Default memory project base
_PROJECTS_DIR = Path.home() / ".hermes" / "memory" / "projects"


def _load_prompt(name: str) -> str:
    """Load a prompt template from the prompts directory."""
    path = _PROMPTS_DIR / f"{name}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _utc_now() -> str:
    """Return current UTC ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _ensure_project_dir(project: str) -> Path:
    """Ensure the project memory directory exists.

    Args:
        project: Project name (e.g. 'hermes-memory').

    Returns:
        Path to the project memory directory.
    """
    project_dir = _PROJECTS_DIR / project
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def _read_existing(path: Path) -> str:
    """Read existing file content, or return empty string if doesn't exist."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _split_above_below_marker(content: str) -> tuple[str, str]:
    """Split content at the auto-generated marker.

    Returns:
        Tuple of (above_marker, below_marker). If marker not found,
        above_marker is the full content, below_marker is empty.
    """
    if _AUTO_GENERATED_MARKER in content:
        parts = content.split(_AUTO_GENERATED_MARKER, 1)
        return parts[0].rstrip("\n"), parts[1]
    return content, ""


def _format_facts(facts: List[Dict[str, Any]]) -> str:
    """Format facts as a markdown bullet list."""
    if not facts:
        return ""
    lines = []
    for fact in facts:
        text = fact.get("fact_text", fact.get("text", ""))
        if not text:
            continue
        scope = fact.get("scope", "general")
        confidence = fact.get("confidence", 0.5)
        project = fact.get("project", "")
        entity = fact.get("entity", "")
        tags = fact.get("tags", [])
        source_ref = fact.get("source_ref", fact.get("source_refs_json", ""))
        # Handle source_refs_json array if it's a JSON string
        if isinstance(source_ref, list) and source_ref:
            source_ref = source_ref[0]
        if isinstance(source_ref, str) and source_ref.startswith("["):
            import json
            try:
                refs = json.loads(source_ref)
                source_ref = refs[0] if refs else ""
            except Exception:
                pass

        lines.append(f"- {text} (scope: {scope}, confidence: {confidence}, source: {source_ref})")
        if project:
            lines.append(f"  - project: {project}")
        if entity:
            lines.append(f"  - entity: {entity}")
        if tags:
            tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
            lines.append(f"  - tags: {tags_str}")
    return "\n".join(lines)


def _format_decisions(decisions: List[Dict[str, Any]]) -> str:
    """Format decisions as a markdown bullet list."""
    if not decisions:
        return ""
    lines = []
    for dec in decisions:
        text = dec.get("decision_text", dec.get("text", ""))
        if not text:
            continue
        rationale = dec.get("rationale", "")
        project = dec.get("project", "")
        owner = dec.get("owner", "")
        source_ref = dec.get("source_ref", "")
        # Handle source_refs_json array
        if isinstance(source_ref, str) and source_ref.startswith("["):
            import json
            try:
                refs = json.loads(source_ref)
                source_ref = refs[0] if refs else ""
            except Exception:
                pass

        lines.append(f"- {text} (source: {source_ref})")
        if rationale:
            lines.append(f"  - rationale: {rationale}")
        if project:
            lines.append(f"  - project: {project}")
        if owner:
            lines.append(f"  - owner: {owner}")
    return "\n".join(lines)


def _format_questions(questions: List[Dict[str, Any]]) -> str:
    """Format open questions as a markdown bullet list."""
    if not questions:
        return ""
    lines = []
    for q in questions:
        text = q.get("question_text", q.get("text", ""))
        if not text:
            continue
        priority = q.get("priority", "medium")
        project = q.get("project", "")
        source_ref = q.get("source_ref", "")
        # Handle source_refs_json array
        if isinstance(source_ref, str) and source_ref.startswith("["):
            import json
            try:
                refs = json.loads(source_ref)
                source_ref = refs[0] if refs else ""
            except Exception:
                pass

        lines.append(f"- {text} (priority: {priority}, source: {source_ref})")
        if project:
            lines.append(f"  - project: {project}")
    return "\n".join(lines)


def _update_single_file(
    file_path: Path,
    section_type: str,
    new_items_text: str,
    dry_run: bool = False,
) -> bool:
    """Update a single project memory file.

    Args:
        file_path: Path to the file.
        section_type: One of 'facts', 'decisions', 'open_questions'.
        new_items_text: Formatted string of new items to append.
        dry_run: If True, compute result but don't write.

    Returns:
        True if file would be updated (or was updated), False if no change needed.
    """
    existing = _read_existing(file_path)
    above, below = _split_above_below_marker(existing)

    if new_items_text.strip():
        new_section = f"\n## {section_type.title()}\n{new_items_text}\n"
        new_content = above.rstrip() + f"\n{_AUTO_GENERATED_MARKER}\n" + new_section
        if below.strip():
            new_content += below.strip() + "\n"
    else:
        # No new items — preserve existing below-marker content
        if existing:
            new_content = above.rstrip() + f"\n{_AUTO_GENERATED_MARKER}\n" + below.lstrip()
        else:
            # No existing file and no new items — nothing to do
            return False

    if file_path.exists() and file_path.read_text(encoding="utf-8") == new_content:
        return False

    if not dry_run:
        file_path.write_text(new_content, encoding="utf-8")
    return True


def _call_llm_render(
    existing_content: str,
    new_facts: str,
    new_decisions: str,
    new_questions: str,
    llm_endpoint: str,
    llm_model: str,
) -> str:
    """Call LLM to render the update_project_memory.md template.

    Args:
        existing_content: Full existing memory.md content.
        new_facts: Formatted facts string.
        new_decisions: Formatted decisions string.
        new_questions: Formatted questions string.
        llm_endpoint: LLM endpoint URL.
        llm_model: Model name.

    Returns:
        Updated markdown content from LLM.

    Raises:
        RuntimeError: If LLM call fails.
    """
    template = _load_prompt("update_project_memory")
    if not template:
        raise RuntimeError("update_project_memory.md template not found")

    # Replace placeholders in template
    user_prompt = template
    user_prompt = user_prompt.replace("{{existing_memory}}", existing_content or "(empty)")
    user_prompt = user_prompt.replace("{{new_facts}}", new_facts or "(none)")
    user_prompt = user_prompt.replace("{{new_decisions}}", new_decisions or "(none)")
    user_prompt = user_prompt.replace("{{new_questions}}", new_questions or "(none)")

    try:
        import requests
        response = requests.post(
            f"{llm_endpoint.rstrip('/')}/chat/completions",
            json={
                "model": llm_model,
                "messages": [
                    {"role": "system", "content": "You are a project memory updater. Output ONLY valid markdown."},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 4096,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip()
    except Exception as exc:
        logger.warning("LLM update_project_memory failed: %s", exc)
        raise RuntimeError(f"LLM render failed: {exc}") from exc


def update_project_memory(
    project: str,
    new_facts: List[Dict[str, Any]],
    new_decisions: List[Dict[str, Any]],
    new_questions: List[Dict[str, Any]],
    llm_endpoint: Optional[str] = "http://192.168.2.105:1234/v1",
    llm_model: str = "qwen3.6-35b-instruct",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Update project memory files with new facts, decisions, and questions.

    This is stage 6 of the dreamer pipeline (TDD §10.1).

    Args:
        project: Project name (e.g. 'hermes-memory').
        new_facts: List of fact dicts from dreamer.
        new_decisions: List of decision dicts from dreamer.
        new_questions: List of question dicts from dreamer.
        llm_endpoint: LLM endpoint for rendering the update_project_memory template.
        llm_model: Model name.
        dry_run: If True, don't write to disk.

    Returns:
        Dict with counts of items updated and files modified.
    """
    project_dir = _ensure_project_dir(project)
    results = {
        "project": project,
        "facts_updated": len(new_facts),
        "decisions_updated": len(new_decisions),
        "questions_updated": len(new_questions),
        "files_modified": [],
    }

    # Read existing content for LLM rendering of memory.md
    memory_path = project_dir / "memory.md"
    existing_memory = _read_existing(memory_path)

    # Format new items as text
    formatted_facts = _format_facts(new_facts)
    formatted_decisions = _format_decisions(new_decisions)
    formatted_questions = _format_questions(new_questions)

    # Try LLM-based merge for memory.md
    if llm_endpoint and not dry_run:
        try:
            llm_content = _call_llm_render(
                existing_memory,
                formatted_facts,
                formatted_decisions,
                formatted_questions,
                llm_endpoint,
                llm_model,
            )
            if llm_content:
                memory_path.write_text(llm_content, encoding="utf-8")
                results["files_modified"].append("memory.md")
        except Exception as exc:
            logger.warning("LLM render for memory.md failed, falling back to append: %s", exc)
            # Fallback: append items below the marker
            _append_below_marker(memory_path, formatted_facts, formatted_decisions, formatted_questions)
            results["files_modified"].append("memory.md")
    elif dry_run:
        # Dry run — just compute what would be written
        pass
    else:
        # No LLM — append below marker
        _append_below_marker(memory_path, formatted_facts, formatted_decisions, formatted_questions)
        results["files_modified"].append("memory.md")

    # Update individual files
    if new_facts:
        facts_path = project_dir / "facts.md"
        if _update_single_file(facts_path, "facts", formatted_facts, dry_run):
            results["files_modified"].append("facts.md")

    if new_decisions:
        decisions_path = project_dir / "decisions.md"
        if _update_single_file(decisions_path, "decisions", formatted_decisions, dry_run):
            results["files_modified"].append("decisions.md")

    if new_questions:
        questions_path = project_dir / "open_questions.md"
        if _update_single_file(questions_path, "open_questions", formatted_questions, dry_run):
            results["files_modified"].append("open_questions.md")

    return results


def _append_below_marker(
    memory_path: Path,
    facts_text: str,
    decisions_text: str,
    questions_text: str,
) -> None:
    """Append new items below the auto-generated marker (fallback when no LLM)."""
    existing = _read_existing(memory_path)
    above, below = _split_above_below_marker(existing)

    new_below_parts = [below.strip()]
    if facts_text:
        new_below_parts.append(f"\n## Facts\n{facts_text}\n")
    if decisions_text:
        new_below_parts.append(f"\n## Decisions\n{decisions_text}\n")
    if questions_text:
        new_below_parts.append(f"\n## Open Questions\n{questions_text}\n")

    new_content = above.rstrip() + f"\n{_AUTO_GENERATED_MARKER}\n" + "\n".join(new_below_parts)
    memory_path.write_text(new_content, encoding="utf-8")


def read_project_memory(project: str) -> Dict[str, str]:
    """Read all project memory files.

    Args:
        project: Project name.

    Returns:
        Dict mapping filename to content for each existing project memory file.
    """
    project_dir = _PROJECTS_DIR / project
    result = {}
    for fname in _PROJECT_MEMORY_FILES:
        path = project_dir / fname
        if path.exists():
            result[fname] = path.read_text(encoding="utf-8")
    return result