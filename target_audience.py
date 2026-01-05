# target_audience.py

from typing import List, Optional
from data_structures import (
    TargetAudience,
    NarrativeFeatures,
    VulnerabilityMap
)


# ---------------------------------------------------------
# Helper: normalize strings
# ---------------------------------------------------------
def _lower_list(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    return [v.lower() for v in values]


# ---------------------------------------------------------
# Psychological vulnerability inference
# ---------------------------------------------------------
def infer_psychological_vulnerabilities(
    features: NarrativeFeatures,
    ta: TargetAudience
) -> List[str]:
    hits: List[str] = []

    # Emotion-based vulnerabilities
    if "fear" in features.emotional_features:
        hits.append("fear-response")
    if "anger" in features.emotional_features:
        hits.append("moral-outrage")
    if "urgency" in features.emotional_features:
        hits.append("scarcity/urgency-bias")

    # Threat-based vulnerabilities
    if "explicit-threat" in features.rhetorical_features:
        hits.append("coercive-threat-susceptibility")

    # Known vulnerabilities from TA
    known = _lower_list(getattr(ta, "known_vulnerabilities", []))
    if any("anxiety" in v for v in known):
        hits.append("general-anxiety")
    if any("economic" in v for v in known):
        hits.append("economic-anxiety")
    if any("distrust" in v for v in known):
        hits.append("institutional-distrust-sensitivity")

    # Deduplicate
    return list(set(hits))


# ---------------------------------------------------------
# Sociocultural vulnerability inference
# ---------------------------------------------------------
def infer_sociocultural_vulnerabilities(
    features: NarrativeFeatures,
    ta: TargetAudience
) -> List[str]:
    hits: List[str] = []

    known = _lower_list(getattr(ta, "known_vulnerabilities", []))
    groups = _lower_list(getattr(ta, "group_identities", []))
    channels = _lower_list(getattr(ta, "information_channels", []))
    orientation = (getattr(ta, "political_orientation", "") or "").lower()

    # Identity-based vulnerabilities
    if "outgroup-threat" in features.identity_features:
        hits.append("identity-polarization")

    # Institutional distrust
    if "delegitimization-of-institutions" in features.rhetorical_features:
        hits.append("institutional-distrust")
    if any("institutional distrust" in v for v in known):
        hits.append("institutional-distrust")

    # Politicized identity
    if orientation in ["left", "right", "far-left", "far-right"]:
        hits.append("politicized-identity-salience")

    # Group-specific vulnerabilities (placeholder logic)
    if any("workers" in g for g in groups):
        hits.append("labor-grievance-susceptibility")
    if any("parents" in g for g in groups):
        hits.append("family-safety-sensitivity")

    # Channel-based vulnerabilities
    if any("facebook" in c or "social" in c for c in channels):
        hits.append("social-media-amplification-risk")
    if any("local news" in c or "radio" in c for c in channels):
        hits.append("local-ecosystem-amplification-risk")

    return list(set(hits))


# ---------------------------------------------------------
# Alignment scoring
# ---------------------------------------------------------
def compute_alignment_score(
    psychological_hits: List[str],
    sociocultural_hits: List[str]
) -> float:
    total = len(psychological_hits) + len(sociocultural_hits)
    max_possible = 8  # tunable; think "how many CVF dimensions we care about"
    if max_possible <= 0:
        return 0.0
    return min(1.0, total / max_possible)


# ---------------------------------------------------------
# Main TA analysis entrypoint
# ---------------------------------------------------------
def analyze_target_audience(
    features: NarrativeFeatures,
    ta: TargetAudience
) -> VulnerabilityMap:

    psychological_hits = infer_psychological_vulnerabilities(features, ta)
    sociocultural_hits = infer_sociocultural_vulnerabilities(features, ta)
    alignment_score = compute_alignment_score(psychological_hits, sociocultural_hits)

    summary = {
        "psychological_hits_count": len(psychological_hits),
        "sociocultural_hits_count": len(sociocultural_hits),
        "note": (
            "High resonance with TA stressors" if alignment_score >= 0.66 else
            "Moderate resonance" if alignment_score >= 0.33 else
            "Low apparent resonance"
        )
    }

    return VulnerabilityMap(
        psychological_hits=psychological_hits,
        sociocultural_hits=sociocultural_hits,
        alignment_score=alignment_score,
        vulnerability_summary=summary
    )
