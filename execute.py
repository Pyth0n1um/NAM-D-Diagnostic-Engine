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
from target_audience import analyze_target_audience
from AI_narrative_extraction import extract_narrative_features
from cvf_graph_viz import visualize_ttp_overlap_network
from cvf_model import map_to_cvf
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


def RISK_SCORER(vuln_map, peripheral, cvf_result):
    """
    Computes a PMESII‑based instability score (0–1)
    and converts it into a 1–100 risk index.
    """

    # --- PMESII DOMAIN INSTABILITY ---

    # Political: identity conflict, institutional distrust
    political = 1.0 if "institutional distrust" in vuln_map.sociocultural_hits else 0.3

    # Military: threat framing
    military = 1.0 if "threat" in peripheral.framing_patterns else 0.2

    # Economic: economic anxiety
    economic = 1.0 if "economic anxiety" in vuln_map.sociocultural_hits else 0.3

    # Social: identity polarization
    social = 0.6 if vuln_map.psychological_hits else 0.2

    # Information: CVF cognitive activation
    information = cvf_result.cvf_score  # already 0–1

    # Infrastructure: scarcity, urgency, crisis windows
    infrastructure = 0.7 if "crisis_window" in peripheral.temporal_cues else 0.3

    # --- Weighted PMESII instability ---
    instability = (
        0.20 * political +
        0.10 * military +
        0.20 * economic +
        0.20 * social +
        0.20 * information +
        0.10 * infrastructure
    )

    instability = max(0.0, min(1.0, instability))

    # --- Convert to 1–100 risk index ---
    risk_index = int(instability * 99) + 1

    return RiskAssessment(
        risk_index=risk_index,
        instability=instability,
        confidence=0.75,
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
def REPORT_GENERATOR(features: NarrativeFeatures,
                     vuln_map: VulnerabilityMap,
                     peripheral: PeripheralSignals,
                     risk: RiskAssessment,
                     ta: TargetAudience) -> DiagnosticReport:
    ta_dict = asdict(ta)

    # Build a human-readable, analyst-style report
    full_text = f"""
NARRATIVE DIAGNOSTIC REPORT
===========================

[1] TARGET AUDIENCE SUMMARY
---------------------------
Demographics:
  - Age Range: {ta_dict.get('demographics', {}).get('age_range')}
  - Location: {ta_dict.get('demographics', {}).get('location')}
  - Education: {ta_dict.get('demographics', {}).get('education_level')}

Orientation:
  - Political Orientation: {ta_dict.get('political_orientation')}
  - Group Identities: {', '.join(ta_dict.get('group_identities', []) or [])}
  - Known Vulnerabilities: {', '.join(ta_dict.get('known_vulnerabilities', []) or [])}
  - Information Channels: {', '.join(ta_dict.get('information_channels', []) or [])}

[2] COGNITIVE VULNERABILITY PROFILE
-----------------------------------
Psychological Vulnerabilities Exploited:
  - {', '.join(vuln_map.psychological_hits) or 'None clearly detected'}

Sociocultural Vulnerabilities Exploited:
  - {', '.join(vuln_map.sociocultural_hits) or 'None clearly detected'}

Summary:
  - {vuln_map.vulnerability_summary.get('note')}

[3] NARRATIVE CHARACTERISTICS
-----------------------------
Narrative Category:
  - {features.narrative_frames[0] if features.narrative_frames else 'Unclear'}  # Updated to use new field

Detected Intents:
  - {', '.join(features.detected_intents) or 'Unclear'}

Emotional Framing:
  - {', '.join(features.emotional_features) or 'None prominent'}

Identity Framing:
  - {', '.join(features.identity_features) or 'None prominent'}

Rhetorical Devices:
  - {', '.join(features.rhetorical_devices) or 'None prominent'}  # Updated to new LLM field

TTP Signals:
  - {', '.join(features.ttp_signals) or 'None detected'}  # New LLM field

Cross-Narrative Links:
  - {', '.join(features.cross_narrative_links) or 'None detected'}  # New LLM field

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

Drivers:
  - Political: {risk.p:.2f}
  - Military (threat framing): {risk.m:.2f}
  - Economic: {risk.e:.2f}
  - Social: {risk.s:.2f}
  - Information (CVF): {risk.i:.2f}
  - Infrastructure (temporal cues): {risk.infra:.2f}
"""

    return DiagnosticReport(
        target_audience_summary=ta_dict,
        psychological_vulnerabilities=vuln_map.psychological_hits,
        sociocultural_vulnerabilities=vuln_map.sociocultural_hits,
        narrative_category=features.narrative_frames[0]
            if features.narrative_frames else "Unclear",  # Updated
        peripheral_indicators={
            "framing_patterns": peripheral.framing_patterns,
            "temporal_cues": peripheral.temporal_cues,
            "cognitive_load": peripheral.cognitive_load,
            "peripheral_score": peripheral.peripheral_score
        },
        risk_score=risk.risk_index,
        full_report_text=full_text.strip()
    )


# -----------------------------
# MAIN EXECUTION WORKFLOW
# -----------------------------
def run_pipeline(narrative_text: str, ta_raw: Dict[str, Any]):
    payload = INPUT_LAYER(narrative_text, ta_raw)
    cleaned = NARRATIVE_INGEST(payload.narrative)

    # 1. LLM-powered narrative feature extraction
    llm_features_dict = extract_narrative_features(cleaned.raw_text)

    # Map LLM output to NarrativeFeatures dataclass
    # We use only the fields your grammar enforces + smart legacy mapping
    features = NarrativeFeatures(
        # Core fields from LLM
        actors=llm_features_dict.get("actors", []),
        claims=llm_features_dict.get("claims", []),
        evidence=llm_features_dict.get("evidence", []),
        sentiment=llm_features_dict.get("sentiment", "neutral"),
        stance=llm_features_dict.get("stance", "neutral"),
        narrative_frames=llm_features_dict.get("narrative_frames", []),
        rhetorical_devices=llm_features_dict.get("rhetorical_devices", []),
        ttp_signals=llm_features_dict.get("ttp_signals", []),
        cross_narrative_links=llm_features_dict.get("cross_narrative_links", []),

        # Legacy field mapping for backward compatibility
        # These ensure REPORT_GENERATOR and other modules don't break
        structural_features=llm_features_dict.get("narrative_frames", []),  # Best proxy: frames are structural
        detected_intents=llm_features_dict.get("claims", []),               # Claims often reveal intent
        emotional_features=llm_features_dict.get("rhetorical_devices", []), # Emotional rhetoric (e.g., fear appeal)
        identity_features=[
            actor for actor in llm_features_dict.get("actors", [])
            if any(identity_keyword in actor.lower() for identity_keyword in 
                   ["regime", "forces", "people", "civilians", "complex", "government"])
        ],  # Rough identity proxy from actors
        rhetorical_features=llm_features_dict.get("rhetorical_devices", [])  # Direct match (old name)
    )

    # 2. Target Audience analysis
    vuln_map = analyze_target_audience(features, payload.target_audience)

    # 3. Peripheral signals
    peripheral = PERIPHERAL_ANALYZER(
        cleaned.raw_text,
        features,
        vuln_map,
        payload.target_audience
    )

    # 4. CVF mapping
    cvf_result = map_to_cvf(
        features,
        payload.target_audience,
        vuln_map
    )

    # 5. Risk scoring
    risk = RISK_SCORER(
        vuln_map,
        peripheral,
        cvf_result=cvf_result
    )

    # 6. CVF graph
    visualize_ttp_overlap_network(cvf_result, threshold=0.6)
    visualize_pmesii_radar(risk)

    # 7. Final report
    report = REPORT_GENERATOR(
        features,
        vuln_map,
        peripheral,
        risk,
        payload.target_audience
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
