#!/usr/bin/env python3
"""Hard gate: fails CI if mutmut's mutation score drops below MIN_SCORE.
Reads mutmut's progress output (emoji counts), not a cache file, since
mutmut run always exits 0 regardless of survivors."""
import re
import sys

MIN_SCORE = 67.0
PATTERN = re.compile(r"\U0001f389\s*(\d+)\s*\U0001fae5\s*(\d+)\s*⏰\s*(\d+)\s*\U0001f914\s*(\d+)\s*\U0001f641\s*(\d+)")


def main() -> int:
    log_path = sys.argv[1]
    text = open(log_path, encoding="utf-8", errors="replace").read()
    matches = PATTERN.findall(text)
    if not matches:
        print("Could not find mutmut's result summary in the log -- did the run crash?")
        return 1

    killed, _no_tests, timeout, suspicious, survived = map(int, matches[-1])
    total = killed + timeout + suspicious + survived
    score = (killed / total * 100) if total else 0.0
    print(f"Mutation score: {score:.2f}% ({killed} killed / {total} meaningful mutants, {survived} survived)")

    if score < MIN_SCORE:
        print(f"FAILED: mutation score {score:.2f}% is below the {MIN_SCORE}% hard gate")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
