from typing import List, Optional, Dict, Any
from data_structures import (
    TargetAudience,
    NarrativeFeatures,
    VulnerabilityMap
)

# ---------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------
def _lower_list(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    return [v.lower() for v in values if v]

def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(kw in text for kw in keywords)

# ---------------------------------------------------------
# Dynamic Psychological Vulnerability Inference
# ---------------------------------------------------------
def infer_psychological_vulnerabilities(
    features: NarrativeFeatures,
    ta: TargetAudience,
    identified_ttps: List[Dict[str, Any]]  # New: TTPs from identification
) -> List[str]:
    hits: List[str] = []

    emotional_lower = " ".join(_lower_list(features.emotional_features))
    rhetorical_lower = " ".join(_lower_list(features.rhetorical_features))
    known_lower = _lower_list(getattr(ta, "known_vulnerabilities", []))

    # Emotion-driven
    if _contains_any(emotional_lower, ["fear", "threat", "danger"]):
        hits.append("fear-response")
    if _contains_any(emotional_lower, ["anger", "outrage", "injustice"]):
        hits.append("moral-outrage")
    if _contains_any(emotional_lower, ["urgency", "scarcity", "limited"]):
        hits.append("scarcity-urgency-bias")

    # Rhetorical coercion
    if _contains_any(rhetorical_lower, ["explicit-threat", "coercion", "force"]):
        hits.append("coercive-threat-susceptibility")

    # TTP-driven psychological amplification
    ttp_names = " ".join([t["name"].lower() for t in identified_ttps])
    if _contains_any(ttp_names, ["fear", "confirmation bias", "scarcity", "social proof"]):
        hits.append("ttp-amplified-cognitive-vulnerability")

    # Known TA vulnerabilities
    if any("anxiety" in v for v in known_lower):
        hits.append("general-anxiety")
    if any("economic" in v for v in known_lower):
        hits.append("economic-anxiety")
    if any("distrust" in v for v in known_lower):
        hits.append("institutional-distrust-sensitivity")

    return list(set(hits))


# ---------------------------------------------------------
# Dynamic Sociocultural Vulnerability Inference
# ---------------------------------------------------------
def infer_sociocultural_vulnerabilities(
    features: NarrativeFeatures,
    ta: TargetAudience,
    identified_ttps: List[Dict[str, Any]]
) -> List[str]:
    hits: List[str] = []

    groups_lower = _lower_list(getattr(ta, "group_identities", []))
    channels_lower = _lower_list(getattr(ta, "information_channels", []))
    orientation = (getattr(ta, "political_orientation", "") or "").lower()
    known_lower = _lower_list(getattr(ta, "known_vulnerabilities", []))

    # Identity polarization
    if any(word in " ".join(features.identity_features).lower() for word in ["outgroup", "enemy", "them vs us"]):
        hits.append("identity-polarization")

    # Institutional distrust
    if any(word in " ".join(features.rhetorical_features).lower() for word in ["corruption", "elite", "deep state", "delegitimization"]):
        hits.append("institutional-distrust")
    if any("distrust" in v for v in known_lower):
        hits.append("institutional-distrust")

    # Politicized identity
    if orientation in ["left", "right", "far-left", "far-right", "extremist"]:
        hits.append("politicized-identity-salience")

    # Group-specific
    if any("workers" in g or "labor" in g for g in groups_lower):
        hits.append("labor-grievance-susceptibility")
    if any("parents" in g or "family" in g for g in groups_lower):
        hits.append("family-safety-sensitivity")
    if any("religious" in g for g in groups_lower):
        hits.append("faith-based-identity-salience")

    # Channel amplification risk
    if any(ch in channels_lower for ch in ["facebook", "tiktok", "youtube", "social media"]):
        hits.append("social-media-amplification-risk")
    if any(ch in channels_lower for ch in ["local news", "radio", "community"]):
        hits.append("local-ecosystem-amplification-risk")

    # TTP-driven sociocultural amplification
    ttp_names = " ".join([t["name"].lower() for t in identified_ttps])
    if _contains_any(ttp_names, ["distort information", "hashtag hijacking", "amplify", "fake experts"]):
        hits.append("disinformation-campaign-susceptibility")

    return list(set(hits))


# ---------------------------------------------------------
# TA Resonance Scoring (0–1)
# ---------------------------------------------------------
def compute_ta_resonance_score(
    psychological_hits: List[str],
    sociocultural_hits: List[str],
    identified_ttps: List[Dict[str, Any]]
) -> float:
    """
    Produces a 0–1 resonance score that can be directly fed into RISK_SCORER.
    Higher = narrative more precisely targets this audience's vulnerabilities.
    """
    base_hits = len(psychological_hits) + len(sociocultural_hits)
    ttp_boost = len(identified_ttps) * 0.15  # Each strong TTP adds weight

    raw_score = base_hits + ttp_boost
    max_possible = 12  # Tunable ceiling (8 hits + 4 strong TTPs)

    return min(1.0, raw_score / max_possible)


# ---------------------------------------------------------
# Main Target Audience Analysis
# ---------------------------------------------------------
def analyze_target_audience(
    features: NarrativeFeatures,
    ta: TargetAudience,
    identified_ttps: List[Dict[str, Any]]  # New required input
) -> VulnerabilityMap:

    psychological_hits = infer_psychological_vulnerabilities(features, ta, identified_ttps)
    sociocultural_hits = infer_sociocultural_vulnerabilities(features, ta, identified_ttps)

    ta_resonance = compute_ta_resonance_score(psychological_hits, sociocultural_hits, identified_ttps)

    summary = {
        "psychological_hits_count": len(psychological_hits),
        "sociocultural_hits_count": len(sociocultural_hits),
        "ttp_amplification_count": len(identified_ttps),
        "ta_resonance_score": round(ta_resonance, 3),
        "note": (
            "HIGH audience resonance — narrative precisely targets known vulnerabilities"
            if ta_resonance >= 0.7 else
            "MODERATE resonance — partial alignment with audience stressors"
            if ta_resonance >= 0.4 else
            "LOW resonance — limited targeting of audience vulnerabilities"
        )
    }

    return VulnerabilityMap(
        psychological_hits=psychological_hits,
        sociocultural_hits=sociocultural_hits,
        alignment_score=ta_resonance,  # Renamed for clarity
        vulnerability_summary=summary
    )
