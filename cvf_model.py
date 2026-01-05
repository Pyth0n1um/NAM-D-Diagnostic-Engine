# cvf_model.py

from dataclasses import dataclass
from typing import List, Dict, Any
from data_structures import NarrativeFeatures, TargetAudience, VulnerabilityMap
from ttp_clustering import cluster_ttp, TTP_DESCRIPTIONS  # Import the expanded descriptions and clustering function


# -----------------------------
# Core CVF / CAT-inspired types
# -----------------------------
@dataclass
class CVFNode:
    id: str
    name: str
    layer: str          # "Vulnerability", "Exploit", "TTP"
    category: str       # e.g., "Trust", "Attention", "Identity", "Framing"
    description: str


@dataclass
class CVFResult:
    activated_vulnerabilities: List[CVFNode]
    activated_exploits: List[CVFNode]
    inferred_ttp_patterns: List[CVFNode]
    dominant_ttp_cluster: Dict[str, Any]  # Result from cluster_ttp
    cvf_score: float
    details: Dict[str, Any]


# -----------------------------
# Dynamic CAT Vulnerability Registry
# -----------------------------
# Automatically build CVF nodes from your expanded TTP_DESCRIPTIONS in ttp_clustering.py
CVF_REGISTRY: Dict[str, CVFNode] = {
    cat_id: CVFNode(
        id=cat_id,
        name=cat_id.replace("CAT-", "").replace("_", " ").title(),
        layer="Vulnerability",  # All current entries are vulnerabilities
        category="Cognitive Bias",  # Can be refined later per CAT layer/category
        description=desc
    )
    for cat_id, desc in TTP_DESCRIPTIONS.items()
}

# Optional: Add known Exploits and TTPs manually if you expand beyond vulnerabilities
# Example:
# CVF_REGISTRY["TTP_FIREHOSE_FALSEHOOD"] = CVFNode(
#     id="TTP_FIREHOSE_FALSEHOOD",
#     name="Firehose of Falsehood",
#     layer="TTP",
#     category="Propagation",
#     description="High-volume, multi-channel, low-fidelity messaging to overwhelm cognition."
# )


# -----------------------------
# Enhanced CVF Mapping Using LLM Features + Clustering
# -----------------------------
def map_to_cvf(
    features: NarrativeFeatures,
    ta: TargetAudience,
    vuln_map: VulnerabilityMap
) -> CVFResult:
    activated_vulns: List[CVFNode] = []
    activated_exploits: List[CVFNode] = []
    inferred_ttps: List[str] = []  # Will collect CAT IDs for clustering

    # --- Activate Vulnerabilities Based on LLM Output ---

    # Sentiment & Stance → Emotional / Identity Vulnerabilities
    if features.sentiment == "negative":
        if "fear" in " ".join(features.rhetorical_devices).lower():
            inferred_ttps.append("CAT-2022-266")  # Fear
        if "anger" in " ".join(features.rhetorical_devices).lower():
            inferred_ttps.append("CAT-2022-321")  # Impulsivity
    if features.stance == "critical":
        inferred_ttps.append("CAT-2023-006")  # Institutional Distrust (Campbell’s Law)

    # Rhetorical Devices → Specific Cognitive Vulnerabilities
    rhetorical_lower = " ".join(features.rhetorical_devices).lower()
    if any(word in rhetorical_lower for word in ["scarcity", "limited", "urgent", "now or never"]):
        inferred_ttps.append("CAT-2022-225")  # Scarcity
    if any(word in rhetorical_lower for word in ["everyone", "consensus", "majority", "trending"]):
        inferred_ttps.append("CAT-2022-226")  # Social Proof
    if any(word in rhetorical_lower for word in ["authority", "expert", "official"]):
        inferred_ttps.append("CAT-2022-216")  # Authority Deference
    if any(word in rhetorical_lower for word in ["loaded", "charged", "extreme language"]):
        inferred_ttps.append("CAT-2022-225")  # Loaded Language (add if in CAT)

    # Narrative Frames → Structural/Identity Vulnerabilities
    frames_lower = " ".join(features.narrative_frames).lower()
    if "victim" in frames_lower or "oppression" in frames_lower:
        inferred_ttps.append("CAT-2022-203")  # Ingroup Bias / Victimhood
    if "corruption" in frames_lower or "elite" in frames_lower:
        inferred_ttps.append("CAT-2023-006")  # Institutional Distrust
    if "hero" in frames_lower or "savior" in frames_lower:
        inferred_ttps.append("CAT-2022-071")  # Halo Effect / Positive Illusion

    # Explicit TTP Signals from LLM
    inferred_ttps.extend(features.ttp_signals)  # Direct CAT IDs if your LLM outputs them

    # Cross-narrative links → Hyperstition, Conspiracy Framing
    links_lower = " ".join(features.cross_narrative_links).lower()
    if "conspiracy" in links_lower:
        inferred_ttps.append("CAT-2022-073")  # Illusory Correlation
    if "self-fulfilling" in links_lower:
        inferred_ttps.append("CAT-2024-010")  # Hyperstition

    # Fold in existing vuln_map hits (psych/socio)
    # Map known strings to CAT IDs if possible
    for hit in vuln_map.psychological_hits + vuln_map.sociocultural_hits:
        hit_lower = hit.lower()
        if "distrust" in hit_lower:
            inferred_ttps.append("CAT-2023-006")
        if "anxiety" in hit_lower:
            inferred_ttps.append("CAT-2022-266")  # Fear
        if "identity" in hit_lower:
            inferred_ttps.append("CAT-2022-203")  # Ingroup Bias

    # Deduplicate TTP IDs
    inferred_ttps = list(set(inferred_ttps))

    # Resolve to CVFNodes
    for ttp_id in inferred_ttps:
        if ttp_id in CVF_REGISTRY:
            node = CVF_REGISTRY[ttp_id]
            if node.layer == "Vulnerability":
                activated_vulns.append(node)
            # Future: add Exploit/TTP layers when registry expanded

    # --- TTP Clustering (The Big Upgrade) ---
    ttp_cluster_result = cluster_ttp(inferred_ttps)

    # --- CVF Score Calculation (Weighted by Cluster Strength) ---
    base_score = (
        0.5 * len(activated_vulns) +
        0.3 * len(activated_exploits) +
        0.2 * len(inferred_ttps)
    )

    # Boost score if strong cluster detected
    cluster_boost = 0.0
    if ttp_cluster_result["top_similarity"] > 0.7:
        cluster_boost = 0.3
    elif ttp_cluster_result["top_similarity"] > 0.5:
        cluster_boost = 0.15

    cvf_score = min(1.0, (base_score / 8.0) + cluster_boost)  # Normalized

    return CVFResult(
        activated_vulnerabilities=activated_vulns,
        activated_exploits=activated_exploits,
        inferred_ttp_patterns=[CVF_REGISTRY[tid] for tid in inferred_ttps if tid in CVF_REGISTRY],
        dominant_ttp_cluster=ttp_cluster_result,
        cvf_score=round(cvf_score, 3),
        details={
            "activated_ttp_ids": inferred_ttps,
            "cluster_summary": {
                "dominant_pattern": ttp_cluster_result["cluster_id"],
                "confidence": ttp_cluster_result["top_similarity"],
                "activated_count": ttp_cluster_result["activated_count"]
            },
            "vulnerability_count": len(activated_vulns)
        }
    )