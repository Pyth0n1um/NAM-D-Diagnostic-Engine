from dataclasses import asdict
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Import data structures
# -----------------------------
from data_structures import (
    Payload,
    Narrative,
    TargetAudience,
    Demographics,
    NarrativeFeatures,
    VulnerabilityMap,
    PeripheralSignals,
    RiskAssessment,
    DiagnosticReport
)

from input_layer import INPUT_LAYER
from target_audience import Optional, analyze_target_audience
from AI_narrative_extraction import extract_narrative_features
from cvf_graph_viz import visualize_ttp_radar
from cvf_model import CVFResult
from ttp_id import identify_ttp_dual
from peripheral_analyzer import PERIPHERAL_ANALYZER


# -----------------------------
# 2. NARRATIVE INGEST
# -----------------------------
def NARRATIVE_INGEST(narrative: Narrative) -> Narrative:
    """
    Light preprocessing + segmentation.
    - Split into sentences/clauses
    - Normalize whitespace
    - Preserve raw for reference
    """
    text = (narrative.raw_text or "").strip()
    # naive segmentation for now; replace with nltk/spacy later
    segments = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    narrative.segments = segments
    return narrative


def RISK_SCORER(
    vuln_map,
    peripheral,
    identified_ttps: Optional[List[Dict[str, Any]]] = None
) -> RiskAssessment:
    """
    Computes PMESII-based instability and converts to 1–100 risk index.
    
    New design:
    - No cvf_result dependency
    - Optional boost from identified CAT TTPs (confidence-weighted)
    - More granular hit detection
    - Clear, maintainable logic
    """
    if identified_ttps is None:
        identified_ttps = []

    # --- Base Domain Scores (0.0–1.0) ---

    # Political: institutional distrust, identity conflict
    political_hits = {"institutional distrust", "corruption", "elite control"}
    political = 1.0 if any(hit in political_hits for hit in vuln_map.sociocultural_hits) else 0.3

    # Military: direct threat framing
    military_hits = {"threat", "attack", "aggression", "invasion"}
    military = 1.0 if any(hit in military_hits for hit in peripheral.framing_patterns) else 0.2

    # Economic: scarcity, anxiety
    economic_hits = {"economic anxiety", "scarcity", "crisis"}
    economic = 1.0 if any(hit in economic_hits for hit in vuln_map.sociocultural_hits) else 0.3

    # Social: polarization, identity tension
    social = 0.8 if vuln_map.psychological_hits else 0.2
    social = max(social, 0.6 if "identity" in " ".join(vuln_map.sociocultural_hits).lower() else social)

    # Information: cognitive activation via identified TTPs
    ttp_conf_sum = sum(t["confidence"] for t in identified_ttps)
    information = min(1.0, ttp_conf_sum / 5.0)  # Normalize: 5 high-confidence TTPs = full

    # Infrastructure: urgency, crisis windows
    infra_hits = {"crisis_window", "urgency", "limited time", "now or never"}
    infrastructure = 0.9 if any(hit in infra_hits for hit in peripheral.temporal_cues) else 0.3

    # --- Weighted Instability ---
    instability = (
        0.20 * political +
        0.15 * military +
        0.15 * economic +
        0.20 * social +
        0.20 * information +   # Strong weight — cognitive is core
        0.10 * infrastructure
    )

    instability = max(0.0, min(1.0, instability))

    # --- Risk Index (1–100) ---
    risk_index = int(instability * 99) + 1

    # --- Confidence in Score ---
    # Higher if multiple strong signals
    signal_count = sum([
        political > 0.5,
        military > 0.5,
        economic > 0.5,
        social > 0.5,
        information > 0.5,
        infrastructure > 0.5
    ])
    confidence = 0.6 + 0.4 * (signal_count / 6.0)  # 0.6–1.0

    return RiskAssessment(
        risk_index=risk_index,
        instability=instability,
        confidence=round(confidence, 2),
        p=political,
        m=military,
        e=economic,
        s=social,
        i=information,
        infra=infrastructure
    )

def visualize_pmesii_radar(risk):
    labels = [
        "Political", "Military", "Economic",
        "Social", "Information", "Infrastructure"
    ]

    values = [
        risk.p,
        risk.m,
        risk.e,
        risk.s,
        risk.i,
        risk.infra
    ]

    # Close the loop
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_yticklabels([])  # cleaner look

    plt.title("PMESII Instability Radar")
    plt.tight_layout()
    plt.show()




# -----------------------------
# 7. REPORT GENERATOR
# -----------------------------
def REPORT_GENERATOR(
    features: NarrativeFeatures,
    vuln_map: VulnerabilityMap,
    peripheral: PeripheralSignals,
    risk: RiskAssessment,
    ta: TargetAudience,
    identified_ttps,
) -> DiagnosticReport:
    """
    Generates a comprehensive analyst-style diagnostic report.
    Now includes direct CAT TTP identification with confidence.
    """
    ta_dict = asdict(ta)

    # --- Top Identified CAT TTPs Section ---
    ttp_section = ""
    if identified_ttps:
        ttp_section += "### TOP IDENTIFIED TTPs (CAT + DISARM)\n\n"
        ttp_section += "| Rank | Source  | TTP Name                     | ID              | Confidence |\n"
        ttp_section += "|------|---------|------------------------------|-----------------|------------|\n"
        for rank, ttp in enumerate(identified_ttps, 1):
            source = ttp.get("source", "CAT")          # "CAT" or "DISARM"
            name = ttp["name"]
            ttp_id = ttp["id"]
            conf = ttp["confidence"]

            # Truncate long names for clean table layout (optional)
            display_name = (name[:22] + "...") if len(name) > 25 else name

            ttp_section += (
                f"| {rank:<4} | {source:<7} | {display_name:<35} | {ttp_id:<15}  | {conf:.3f}     |\n"
            )
    else:
        ttp_section += "### TOP IDENTIFIED TTPs (CAT + DISARM)\n"
        ttp_section += "No relevant TTPs identified in the narrative.\n"

    ttp_section += "\n"

    # --- Full Report Text ---
    full_text = f"""
NARRATIVE DIAGNOSTIC REPORT
===========================

[1] TARGET AUDIENCE SUMMARY
---------------------------
Demographics:
  - Age Range: {ta_dict.get('demographics', {}).get('age_range', 'Unknown')}
  - Location: {ta_dict.get('demographics', {}).get('location', 'Unknown')}
  - Education: {ta_dict.get('demographics', {}).get('education_level', 'Unknown')}

Orientation:
  - Political Orientation: {ta_dict.get('political_orientation', 'Unknown')}
  - Group Identities: {', '.join(ta_dict.get('group_identities', []) or ['None'])}
  - Known Vulnerabilities: {', '.join(ta_dict.get('known_vulnerabilities', []) or ['None'])}
  - Information Channels: {', '.join(ta_dict.get('information_channels', []) or ['None'])}

[2] COGNITIVE VULNERABILITY PROFILE
-----------------------------------
Psychological Vulnerabilities Exploited:
  - {', '.join(vuln_map.psychological_hits) or 'None clearly detected'}

Sociocultural Vulnerabilities Exploited:
  - {', '.join(vuln_map.sociocultural_hits) or 'None clearly detected'}

Summary:
  {vuln_map.vulnerability_summary.get('note', 'No summary available')}

[3] NARRATIVE CHARACTERISTICS
-----------------------------
Narrative Category:
  - {features.narrative_frames[0] if features.narrative_frames else 'Unclear'}

Detected Intents:
  - {', '.join(features.detected_intents) or 'Unclear'}

Emotional Framing:
  - {', '.join(features.emotional_features) or 'None prominent'}

Identity Framing:
  - {', '.join(features.identity_features) or 'None prominent'}

Rhetorical Devices:
  - {', '.join(features.rhetorical_devices) or 'None prominent'}

Cross-Narrative Links:
  - {', '.join(features.cross_narrative_links) or 'None detected'}

{ttp_section}

[4] PERIPHERAL INDICATORS (COGNITIVE CONTEXT)
---------------------------------------------
Cognitive Load:
  - {peripheral.cognitive_load:.2f}

Framing Patterns:
  - {', '.join(peripheral.framing_patterns) or 'None detected'}

Temporal Cues:
  - {', '.join(peripheral.temporal_cues) or 'None detected'}

Peripheral Influence Score:
  - {peripheral.peripheral_score:.2f}

[5] RISK ASSESSMENT (PMESII Stability Model)
-------------------------------------------
Risk Index:
  - {risk.risk_index} / 100

Instability (0–1):
  - {risk.instability:.2f}

Domain Drivers:
  - Political: {risk.p:.2f}
  - Military: {risk.m:.2f}
  - Economic: {risk.e:.2f}
  - Social: {risk.s:.2f}
  - Information: {risk.i:.2f}
  - Infrastructure: {risk.infra:.2f}
"""

    return DiagnosticReport(
        target_audience_summary=ta_dict,
        psychological_vulnerabilities=vuln_map.psychological_hits,
        sociocultural_vulnerabilities=vuln_map.sociocultural_hits,
        narrative_category=features.narrative_frames[0] if features.narrative_frames else "Unclear",
        peripheral_indicators={
            "framing_patterns": peripheral.framing_patterns,
            "temporal_cues": peripheral.temporal_cues,
            "cognitive_load": peripheral.cognitive_load,
            "peripheral_score": peripheral.peripheral_score
        },
        risk_score=risk.risk_index,
        full_report_text=full_text.strip(),
        identified_ttps=identified_ttps  # ← Now accepted
    )


# -----------------------------
# MAIN EXECUTION WORKFLOW
# -----------------------------
def run_pipeline(narrative_text: str, ta_raw: Dict[str, Any]):
    """
    Full NAM-D pipeline — local, deterministic, no grammar/LLM server for TTPs.
    - Narrative feature extraction (LLM-powered)
    - Target audience vulnerability analysis
    - Peripheral signals
    - Direct TTP identification via embedding similarity
    - Risk scoring
    - Radar visualizations
    - Final report
    """
    # 1. Input processing
    payload = INPUT_LAYER(narrative_text, ta_raw)
    cleaned = NARRATIVE_INGEST(payload.narrative)

    # 2. LLM-powered narrative feature extraction
    llm_features_dict = extract_narrative_features(cleaned.raw_text)

    # 3. Map LLM output to NarrativeFeatures dataclass
    features = NarrativeFeatures(
        actors=llm_features_dict.get("actors", []),
        claims=llm_features_dict.get("claims", []),
        evidence=llm_features_dict.get("evidence", []),
        sentiment=llm_features_dict.get("sentiment", "neutral"),
        stance=llm_features_dict.get("stance", "neutral"),
        narrative_frames=llm_features_dict.get("narrative_frames", []),
        rhetorical_devices=llm_features_dict.get("rhetorical_devices", []),
        ttp_signals=llm_features_dict.get("ttp_signals", []),  # Legacy — can be empty now
        cross_narrative_links=llm_features_dict.get("cross_narrative_links", []),

        # Legacy mapping (kept for backward compatibility with REPORT_GENERATOR)
        structural_features=llm_features_dict.get("narrative_frames", []),
        detected_intents=llm_features_dict.get("claims", []),
        emotional_features=llm_features_dict.get("rhetorical_devices", []),
        identity_features=[
            actor for actor in llm_features_dict.get("actors", [])
            if any(kw in actor.lower() for kw in ["regime", "forces", "people", "civilians", "government", "elite", "authority"])
        ],
        rhetorical_features=llm_features_dict.get("rhetorical_devices", [])
    )

    # 4. Target audience vulnerability analysis
    vuln_map = analyze_target_audience(features, payload.target_audience)

    # 5. Peripheral signals analysis
    peripheral = PERIPHERAL_ANALYZER(
        cleaned.raw_text,
        features,
        vuln_map,
        payload.target_audience
    )

    # 6. Direct Local TTP Identification (Core of New Pipeline)
    ttp_results = identify_ttp_dual(cleaned.raw_text, top_k_per_registry=5)

    # Create lightweight result container (replaces old cvf_result)
    pipeline_result={
        "features": features,
        "vuln_map": vuln_map,
        "peripheral": peripheral,
        "identified_ttps": ttp_results,
        "ttp_count": len(ttp_results),
        "dominant_ttp": ttp_results[0] if ttp_results else None,
        "activated_ttp_ids": [t["id"] for t in ttp_results],
        "details": {  # For compatibility with old code if needed
            "identified_ttps": ttp_results,
            "activated_ttp_ids": [t["id"] for t in ttp_results]
        }
    }

    # 7. Risk scoring (PMESII-based)
    # Update RISK_SCORER to accept identified_ttps or vuln_map/peripheral only
    risk = RISK_SCORER(vuln_map, peripheral, identified_ttps=ttp_results)

    # 8. Visualizations
    
    visualize_ttp_radar(CVFResult, cleaned.raw_text)  # Uses identify_ttp internally

    visualize_pmesii_radar(risk)

    # 9. Final analyst report
    # Update REPORT_GENERATOR to accept identified_ttps
    report = REPORT_GENERATOR(
        features=features,
        vuln_map=vuln_map,
        peripheral=peripheral,
        risk=risk,
        ta=payload.target_audience,
        identified_ttps=ttp_results  # New parameter
    )

    return report


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    narrative = input("What is the raw text of your narrative: ")
    ta = {
        "demographics": {
            "age_range": input("Enter Target Audience age range: "),
            "location": input("Enter Target Audience location: "),
            "education_level": input("Enter Target Audience education level: ")
        },
        "political_orientation": input("Enter Target Audience political leaning: "),
        "group_identities": ["union workers", "young parents"],
        "known_vulnerabilities": ["economic anxiety", "institutional distrust"],
        "information_channels": ["Facebook", "local news"]
    }

    result = run_pipeline(narrative, ta)
    print(result.full_report_text)
