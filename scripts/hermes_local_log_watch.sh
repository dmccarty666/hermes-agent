#!/usr/bin/env bash
# hermes_local_log_watch.sh
#
# Tail Hermes logs with filters tuned for memory-provider events. Color-
# highlights errors and "good signal" lines for the hermes-local switch.
#
# Usage:
#   bash scripts/hermes_local_log_watch.sh                 # follow agent.log
#   bash scripts/hermes_local_log_watch.sh errors          # follow errors.log
#   bash scripts/hermes_local_log_watch.sh --session <id>  # session filter
#
# All extra args are forwarded to `hermes logs`.

set -uo pipefail

LOG_NAME="${1:-agent}"
shift 2>/dev/null || true

# Patterns
GOOD='Memory provider .*initialized|hermes-local.*registered|hermes-local.*Initialized|on_session_switch|narrative thread|Dream run complete|MemoryStore connection closed|Qdrant.*connected|chunks_indexed_24h|facts_total'
WARN='Qdrant.*offline|Qdrant.*unreachable|LMS.*unreachable|connection refused|embedding.*unavailable|degraded|fallback'
BAD='ERROR|CRITICAL|Traceback|fts5_search:.*failed|no such column|memory provider.*failed|handle_tool_call.*raised|sync_turn failed|prefetch.*failed'

# Filter: only show lines containing a memory-relevant token.
# We pipe `hermes logs --follow` through grep + awk for coloring.
KEEP='memory|hermes-local|holographic|Qdrant|LMS|dream|FTS5|fts5|MemoryStore|narrative|prefetch|sync_turn|memory_query|memory_write|memory_recent_context'

echo "── hermes-local log watch (log=$LOG_NAME) ──────────────────────────────"
echo "  Good signals  (green):  initialized, narrative thread, Dream run complete"
echo "  Warnings      (yellow): Qdrant/LMS offline, fallback, degraded"
echo "  Errors        (red):    ERROR/CRITICAL/Traceback/fts5_search failed"
echo "  Filter keeps lines matching: $KEEP"
echo "─────────────────────────────────────────────────────────────────────────"

hermes logs "$LOG_NAME" --follow "$@" 2>&1 \
| grep -E --line-buffered -i "$KEEP" \
| awk -v G="$GOOD" -v W="$WARN" -v B="$BAD" '
    BEGIN {
        red="\033[1;31m"; ylw="\033[1;33m"; grn="\033[1;32m"; rst="\033[0m"
    }
    {
        if (match($0, B))      printf "%s%s%s\n", red, $0, rst
        else if (match($0, W)) printf "%s%s%s\n", ylw, $0, rst
        else if (match($0, G)) printf "%s%s%s\n", grn, $0, rst
        else                   print $0
        fflush()
    }
'
