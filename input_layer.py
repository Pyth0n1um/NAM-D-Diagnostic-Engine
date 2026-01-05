from data_structures import (
    Payload,
    Narrative,
    TargetAudience,
    Demographics,
    TrustLandscape
)

def normalize_string(value):
    if value is None:
        return None
    return str(value).strip()

def normalize_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    return [str(value).strip()]

def normalize_optional_field(value):
    if value is None:
        return None
    if isinstance(value, list):
        return normalize_list(value)
    if isinstance(value, dict):
        return value
    return normalize_string(value)


def INPUT_LAYER(narrative_text: str, ta_raw: dict) -> Payload:
    """
    Takes raw narrative text and a structured TA dictionary,
    validates and normalizes them, and returns a Payload object.
    """

    # -----------------------------
    # 1. Validate narrative input
    # -----------------------------
    if not narrative_text or not narrative_text.strip():
        raise ValueError("Narrative text is required")

    narrative = Narrative(
        raw_text=narrative_text.strip(),
        metadata={"source": ta_raw.get("source", "unknown")}
    )

    # -----------------------------
    # 2. Validate TA core fields
    # -----------------------------
    required_core = [
        "demographics",
        "political_orientation",
        "group_identities",
        "known_vulnerabilities",
        "information_channels"
    ]

    for field in required_core:
        if field not in ta_raw:
            raise ValueError(f"Missing required TA field: {field}")

    # -----------------------------
    # 3. Normalize demographics
    # -----------------------------
    demo_raw = ta_raw["demographics"]

    demographics = Demographics(
        age_range=normalize_string(demo_raw.get("age_range")),
        location=normalize_string(demo_raw.get("location")),
        education_level=normalize_string(demo_raw.get("education_level"))
    )

    # -----------------------------
    # 4. Normalize optional trust landscape
    # -----------------------------
    trust_raw = ta_raw.get("trust_landscape")
    trust_landscape = None

    if trust_raw:
        trust_landscape = TrustLandscape(
            trusted_institutions=normalize_list(trust_raw.get("trusted_institutions")),
            distrusted_institutions=normalize_list(trust_raw.get("distrusted_institutions"))
        )

    # -----------------------------
    # 5. Build TargetAudience object
    # -----------------------------
    target_audience = TargetAudience(
        demographics=demographics,
        political_orientation=normalize_string(ta_raw["political_orientation"]),
        group_identities=normalize_list(ta_raw["group_identities"]),
        known_vulnerabilities=normalize_list(ta_raw["known_vulnerabilities"]),
        information_channels=normalize_list(ta_raw["information_channels"]),

        # Optional fields
        cultural_context=normalize_optional_field(ta_raw.get("cultural_context")),
        psychological_markers=normalize_optional_field(ta_raw.get("psychological_markers")),
        current_stressors=normalize_optional_field(ta_raw.get("current_stressors")),
        historical_grievances=normalize_optional_field(ta_raw.get("historical_grievances")),
        trust_landscape=trust_landscape,
        media_literacy_level=normalize_optional_field(ta_raw.get("media_literacy_level")),
        community_structure=normalize_optional_field(ta_raw.get("community_structure")),
        polarization_vectors=normalize_optional_field(ta_raw.get("polarization_vectors"))
    )

    # -----------------------------
    # 6. Construct payload
    # -----------------------------
    return Payload(
        narrative=narrative,
        target_audience=target_audience
    )