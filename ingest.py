# ingest.py

from typing import Any, Dict

# Import error types for structured handling
class IngestError(Exception):
    """Base class for ingestion-related errors."""
    pass

class ValidationError(IngestError):
    """Raised when incoming payload fails validation."""
    pass


# -----------------------------
# 1. VALIDATION HELPERS
# -----------------------------

def require_field(payload: Dict[str, Any], field: str):
    if field not in payload:
        raise ValidationError(f"Missing required field: '{field}'")
    return payload[field]

def require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"Field '{field_name}' must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValidationError(f"Field '{field_name}' cannot be empty")
    return cleaned

def require_dict(value: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"Field '{field_name}' must be an object")
    return value

def require_list(value: Any, field_name: str) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    # Convert singletons to list
    return [value]


# -----------------------------
# 2. NORMALIZATION HELPERS
# -----------------------------

def normalize_string(value: Any) -> str:
    if value is None:
        return None
    return str(value).strip()

def normalize_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    return [str(value).strip()]


# -----------------------------
# 3. INGESTION LOGIC
# -----------------------------

def ingest(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and normalizes raw incoming JSON from the client.
    Returns a dict with:
        {
            "narrative_text": str,
            "ta_raw": dict
        }
    This output is passed directly into INPUT_LAYER().
    """

    if not isinstance(payload, dict):
        raise ValidationError("Payload must be a JSON object")

    # -----------------------------
    # A. Validate narrative
    # -----------------------------
    raw_narrative = require_field(payload, "narrative")
    narrative_text = require_string(raw_narrative, "narrative")

    # -----------------------------
    # B. Validate TA block
    # -----------------------------
    raw_ta = require_field(payload, "target_audience")
    ta_raw = require_dict(raw_ta, "target_audience")

    # Required TA fields (must match INPUT_LAYER expectations)
    required_ta_fields = [
        "demographics",
        "political_orientation",
        "group_identities",
        "known_vulnerabilities",
        "information_channels"
    ]

    for field in required_ta_fields:
        if field not in ta_raw:
            raise ValidationError(f"Missing required TA field: '{field}'")

    # -----------------------------
    # C. Normalize TA fields
    # -----------------------------
    # Demographics must be a dict
    ta_raw["demographics"] = require_dict(
        ta_raw["demographics"], "demographics"
    )

    # Lists
    ta_raw["group_identities"] = normalize_list(ta_raw["group_identities"])
    ta_raw["known_vulnerabilities"] = normalize_list(ta_raw["known_vulnerabilities"])
    ta_raw["information_channels"] = normalize_list(ta_raw["information_channels"])

    # Strings
    ta_raw["political_orientation"] = normalize_string(ta_raw["political_orientation"])

    # Optional fields (pass-through)
    optional_fields = [
        "cultural_context",
        "psychological_markers",
        "current_stressors",
        "historical_grievances",
        "trust_landscape",
        "media_literacy_level",
        "community_structure",
        "polarization_vectors"
    ]

    for field in optional_fields:
        if field in ta_raw and ta_raw[field] is None:
            del ta_raw[field]

    # -----------------------------
    # D. Return clean structure
    # -----------------------------
    return {
        "narrative_text": narrative_text,
        "ta_raw": ta_raw
    }
