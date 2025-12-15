import json
import re
from typing import Any, List


def safe_json_loads(text: Any) -> Any:
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return {}
    s = text.strip()
    try:
        blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", s)
        if blocks:
            s = blocks[-1].strip()
    except Exception:
        pass
    try:
        return json.loads(s)
    except Exception:
        pass
    s = s.replace("\r", "\n")
    s = s.replace("'", '"')
    s = re.sub(r",\s*(\]|\})", r"\1", s)
    def _quote_keys(m: re.Match) -> str:
        return f"{m.group(1)}\"{m.group(2)}\"{m.group(3)}"
    s = re.sub(r"([\{,\s])([A-Za-z_][A-Za-z0-9_]*)\s*(:)", _quote_keys, s)
    try:
        return json.loads(s)
    except Exception:
        return [] if s.startswith('[') else {}


def parse_json_array_or_csv(text: Any) -> List[str]:
    data = safe_json_loads(text)
    if isinstance(data, list):
        return [str(x).strip() for x in data if x is not None]
    if isinstance(text, str):
        return [x.strip() for x in text.split(',') if x.strip()]
    return []
