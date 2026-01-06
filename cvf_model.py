from dataclasses import dataclass
from typing import List, Dict, Any
from data_structures import NarrativeFeatures, TargetAudience, VulnerabilityMap
from ttp_id import identify_ttp_dual, TTP_DESCRIPTIONS  # Local identification function

# -----------------------------
# Core CVF / CAT-inspired types
# -----------------------------
@dataclass
class CVFNode:
    id: str
    name: str
    description: str
    confidence: float  # 0.0–1.0 from identification


@dataclass
class CVFResult:
    identified_ttps: List[CVFNode]        # Top identified with confidence
    cvf_score: float                      # Overall cognitive vulnerability score
    details: Dict[str, Any]               # For radar, reporting, etc.


# -----------------------------
# CVF Mapping Using Direct Local Identification
# -----------------------------
def map_to_cvf(
    features: NarrativeFeatures,
    ta: TargetAudience,
    vuln_map: VulnerabilityMap,
    narrative_text: str  # Required for local identification
) -> CVFResult:
    """
    Maps the narrative to CAT TTPs using local embedding-based identification.
    No clustering — direct, confidence-weighted top matches.
    """
    # Step 1: Direct TTP identification from narrative
    identified = identify_ttp_dual(narrative_text, top_k=10, min_confidence=0.3)

    # Convert to CVFNode
    cvf_nodes = []
    total_confidence = 0.0
    for item in identified:
        cvf_nodes.append(CVFNode(
            id=item["id"],
            name=item["name"],
            description=TTP_DESCRIPTIONS.get(item["id"], {}).get("description", "No description"),
            confidence=item["confidence"]
        ))
        total_confidence += item["confidence"]

    # Step 2: CVF Score (simple but effective)
    # Base: number of identified TTPs
    # Boost: average confidence
    count_score = len(cvf_nodes) / 10.0  # Max 10 → 1.0
    avg_confidence = total_confidence / len(cvf_nodes) if cvf_nodes else 0.0
    confidence_boost = avg_confidence * 0.5

    cvf_score = min(1.0, count_score + confidence_boost)
    cvf_score = round(cvf_score, 3)

    # Step 3: Details for downstream use (radar, reporting)
    details = {
        "identified_ttp_count": len(cvf_nodes),
        "average_confidence": round(avg_confidence, 3) if cvf_nodes else 0.0,
        "top_confidence": cvf_nodes[0].confidence if cvf_nodes else 0.0,
        "dominant_ttp": cvf_nodes[0] if cvf_nodes else None
    }

    return CVFResult(
        identified_ttps=cvf_nodes,
        cvf_score=cvf_score,
        details=details
    )
