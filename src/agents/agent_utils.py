import re
import time
import json


def handle_rate_limit(error: Exception) -> int:
    """
    Handle rate limit errors from OpenRouter or Groq.
    Extracts wait time if available, otherwise uses default backoff.
    Returns seconds waited.
    """
    message = str(error)

    # Groq format: "try again in 13m12.287s"
    match = re.search(r'try again in (\d+)m(\d+\.?\d*)', message)
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        wait = int(minutes * 60 + seconds) + 5
    else:
        # OpenRouter format: retry shortly — use exponential backoff
        wait = 30

    print(f"  Rate limit hit — waiting {wait}s before retry...")
    time.sleep(wait)
    return wait


def parse_json_robust(raw: str) -> dict:
    """
    Parse JSON with multiple fallback strategies.
    Handles: direct JSON, markdown fences, JSON embedded in text,
    and unterminated strings by truncating at last valid position.
    """
    raw = raw.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue

    # Strategy 3: find JSON object within surrounding text
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4: truncate at last closing brace
    # handles unterminated strings from long outputs
    last_brace = raw.rfind('}')
    if last_brace > 0:
        truncated = raw[:last_brace + 1]
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass

    # All strategies failed
    raise json.JSONDecodeError("Could not parse JSON from response", raw, 0)