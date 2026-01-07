from dataclasses import asdict
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from cognitive_vulnerability_diagnosis import diagnose_cognitive_vulnerabilities


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

from input_layer import INPUT_LAYER, TargetAudience
from target_audience import Optional, analyze_target_audience
from AI_narrative_extraction import extract_narrative_features
from cvf_graph_viz import visualize_ttp_radar
from viz_cog_vuln import visualize_cog_vuln_radar
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
    vuln_map: VulnerabilityMap,
    peripheral: PeripheralSignals,
    identified_ttps: List[Dict[str, Any]],
    narrative_text: str  # New: for keyword scanning
) -> RiskAssessment:
    """
    Computes PMESII-based instability and converts to 1–100 risk index.
    Fully integrated with dynamic target audience resonance, dual-registry TTPs, and keyword assistance.
    """

    # Base resonance from target audience analysis (0–1)
    ta_resonance = vuln_map.alignment_score

    # TTP confidence lookup by keyword for domain boosts
    ttp_conf_by_keyword = {}
    for t in identified_ttps:
        name_lower = t["name"].lower()
        conf = t["confidence"]
        for kw in name_lower.split():
            ttp_conf_by_keyword[kw] = max(ttp_conf_by_keyword.get(kw, 0.0), conf)

    # Keyword Dictionaries for Narrative Scanning
    political_keywords = ["eu", "un", "icc", "icj", "brussels", "the hague", "brics", "nato", "wto", "g7", "g20", "sanctions", "diplomacy", "treaty", "summit", "election", "regime", "elite", "corruption", "deep state", "polarization", "sovereignty", "international law"]
    military_keywords = ["nato", "army", "navy", "air force", "military equipment", "troops", "battalion", "division", "weapon", "missile", "drone", "cyberwar", "hybrid warfare", "aggression", "invasion", "defense", "alliance", "peacekeeping", "arms race"]
    economic_keywords = ["oil", "opec", "sanctions", "shipping", "gas", "energy", "trade war", "tariff", "currency", "inflation", "recession", "supply chain", "blockade", "boycott", "market crash", "poverty", "wealth gap"]
    social_keywords = ["community", "tribe", "ethnicity", "religion", "protest", "riot", "migration", "demographic", "inequality", "social media amplification", "echo chamber", "genocide", "human rights", "civil unrest", "polarization", "identity politics"]
    information_keywords = ["fake news", "disinformation", "misinformation", "deepfake", "bot", "amplify", "narrative", "framing", "censorship", "media bias", "information warfare"]
    infrastructure_keywords = ["power grid", "transportation", "water", "communication network", "cyber infrastructure", "supply lines", "port", "pipeline", "critical infrastructure"]

    # Scan Narrative for Keyword Hits
    narrative_lower = narrative_text.lower()

    def keyword_hit_score(keywords: List[str], multiplier: float = 0.1) -> float:
        hits = sum(1 for kw in keywords if kw.lower() in narrative_lower)
        return min(hits * multiplier, 0.6)  # Cap to avoid over-boost

    # Domain Boost Function (Unchanged)
    def domain_boost(base: float, keywords: List[str], source_hits: List[str] = None, max_boost: float = 0.5) -> float:
        strength = 0.0
        if source_hits:
            strength += sum(1 for hit in source_hits if any(kw in hit.lower() for kw in keywords)) * 0.2
        strength += sum(ttp_conf_by_keyword.get(kw, 0.0) for kw in keywords) * 0.3
        return min(base + strength, base + max_boost)

    # Political
    political = domain_boost(0.3, political_keywords, vuln_map.sociocultural_hits, max_boost=0.7)
    political += keyword_hit_score(political_keywords, 0.15)
    if any(kw in " ".join([t["name"].lower() for t in identified_ttps]) for kw in ["politicized", "distrust", "identity"]):
        political = min(1.0, political + 0.2)

    # Military
    military = domain_boost(0.35, military_keywords, peripheral.framing_patterns, max_boost=0.65)
    military += keyword_hit_score(military_keywords, 0.12)

    # Economic
    economic = domain_boost(0.3, economic_keywords, vuln_map.sociocultural_hits, max_boost=0.7)
    economic += keyword_hit_score(economic_keywords, 0.15)

    # Social
    social = domain_boost(0.2, social_keywords, vuln_map.psychological_hits + vuln_map.sociocultural_hits, max_boost=0.4)
    social += keyword_hit_score(social_keywords, 0.1)

    # Information (core driver)
    information = min(1.0, sum(t["confidence"] for t in identified_ttps) / 4.0)
    information += keyword_hit_score(information_keywords, 0.18)

    # Infrastructure
    infrastructure = domain_boost(0.3, infrastructure_keywords, peripheral.temporal_cues, max_boost=0.6)
    infrastructure += keyword_hit_score(infrastructure_keywords, 0.12)

    # Clamp to 0–1
    political, military, economic, social, information, infrastructure = [
        max(0.0, min(1.0, v)) for v in [political, military, economic, social, information, infrastructure]
    ]

    # Apply audience resonance multiplier
    instability_components = [
        political * (1 + ta_resonance * 0.2),
        military * (1 + ta_resonance * 0.2),
        economic * (1 + ta_resonance * 0.2),
        social * (1 + ta_resonance * 0.2),
        information * (1 + ta_resonance * 0.3),
        infrastructure * (1 + ta_resonance * 0.15)
    ]

    # Weighted instability
    instability = (
        0.20 * instability_components[0] +
        0.10 * instability_components[1] +
        0.15 * instability_components[2] +
        0.20 * instability_components[3] +
        0.25 * instability_components[4] +
        0.10 * instability_components[5]
    )

    instability = max(0.0, min(1.0, instability))

    # Risk index
    risk_index = int(instability * 99) + 1

    # Confidence
    signal_count = sum([
        political > 0.5,
        military > 0.5,
        economic > 0.5,
        social > 0.5,
        information > 0.5,
        infrastructure > 0.5
    ])
    confidence = round(0.6 + 0.4 * (signal_count / 6.0), 2)

    return RiskAssessment(
        risk_index=risk_index,
        instability=instability,
        confidence=confidence,
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
    cognitive_vulnerabilities: List[Dict[str, Any]]
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
        ttp_section += "### TOP IDENTIFIED TTPs (DISARM)\n"
        ttp_section += "No relevant TTPs identified in the narrative.\n"

    ttp_section += "\n"

    cog_vuln_section = "### COGNITIVE VULNERABILITY PROFILE\n"
    if cognitive_vulnerabilities:
        cog_vuln_section += "| Rank | Vulnerability                       | Confidence          | Key Indicators Matched |\n"
        cog_vuln_section += "|------|-------------------------------------|---------------------|------------------------|\n"
        for rank, vuln in enumerate(cognitive_vulnerabilities, 1):
            name = vuln["name"]
            conf = vuln["confidence"]
            indicators = ", ".join(vuln.get("indicators_matched", [])) or "None explicit"
            cog_vuln_section += f"| {rank:<4} | {name:<35} | {conf:.3f}    | {indicators:<15}              |\n"
        cog_vuln_section += "\n"
        dominant = cognitive_vulnerabilities[0]["name"]
        avg_conf = sum(v["confidence"] for v in cognitive_vulnerabilities) / len(cognitive_vulnerabilities)
        cog_vuln_section += f"**Dominant Vulnerability**: {dominant} (Avg Confidence: {avg_conf:.3f})\n"
    else:
        cog_vuln_section += "No significant cognitive vulnerabilities diagnosed.\n"

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

{cog_vuln_section}

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
    Full NAM-D pipeline — local, deterministic, hybrid CAT+DISARM TTP identification.
    - Narrative feature extraction (LLM-powered)
    - Dynamic target audience vulnerability analysis (TTP-aware)
    - Peripheral signals
    - Dual-registry TTP identification (CAT + DISARM)
    - Risk scoring with audience resonance
    - Radar visualizations
    - Final analyst report
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
        ttp_signals=llm_features_dict.get("ttp_signals", []),  # Legacy
        cross_narrative_links=llm_features_dict.get("cross_narrative_links", []),

        # Legacy compatibility fields
        structural_features=llm_features_dict.get("narrative_frames", []),
        detected_intents=llm_features_dict.get("claims", []),
        emotional_features=llm_features_dict.get("rhetorical_devices", []),
        identity_features=[
            actor for actor in llm_features_dict.get("actors", [])
            if any(kw in actor.lower() for kw in ["regime", "forces", "people", "civilians", "government", "elite", "authority"])
        ],
        rhetorical_features=llm_features_dict.get("rhetorical_devices", [])
    )

    # 4. Dual-registry TTP Identification (CAT + DISARM)
    ttp_results = identify_ttp_dual(cleaned.raw_text, top_k_per_registry=5)

    # 5. Dynamic Target Audience Vulnerability Analysis (uses TTPs)
    ta_vuln_map = analyze_target_audience(features, payload.target_audience, ttp_results)

    # 6. Peripheral signals analysis
    peripheral = PERIPHERAL_ANALYZER(
        cleaned.raw_text,
        features,
        ta_vuln_map,
        payload.target_audience
    )

    # 7. Risk scoring (uses dynamic vuln_map with audience resonance)
    risk = RISK_SCORER(ta_vuln_map, peripheral, ttp_results, cleaned.raw_text)

    # 8. Visualizations
    visualize_ttp_radar(CVFResult, cleaned.raw_text)  # Uses identify_ttp_dual internally
    visualize_pmesii_radar(risk)

    # Convert VulnerabilityMap to dict for profile
    ta_profile_dict = {
        "psychological_hits": ta_vuln_map.psychological_hits,
        "sociocultural_hits": ta_vuln_map.sociocultural_hits,
        "alignment_score": ta_vuln_map.alignment_score,
        "vulnerability_summary": ta_vuln_map.vulnerability_summary
    }

    cognitive_vulnerabilities = diagnose_cognitive_vulnerabilities(
        cleaned.raw_text,
        ta_profile_dict,
        top_k=10
    )

    visualize_cog_vuln_radar(cleaned.raw_text, ta_profile_dict)


    # 9. Final analyst report
    report = REPORT_GENERATOR(
        features=features,
        vuln_map=ta_vuln_map,           # Dynamic version
        peripheral=peripheral,
        risk=risk,
        ta=payload.target_audience,
        identified_ttps=ttp_results,
        cognitive_vulnerabilities=cognitive_vulnerabilities
    )

    # Optional: lightweight result container for future use
    pipeline_result = {
        "features": features,
        "target_audience": payload.target_audience,
        "vuln_map": ta_vuln_map,
        "peripheral": peripheral,
        "identified_ttps": ttp_results,
        "risk": risk,
        "report": report
    }

    return report  # or return pipeline_result if you want more data


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
        "political_orientation": ["Republican-leaning", "conservative"],
        "group_identities": ["union workers", "young parents", "college students", "minority communities"],
        "known_vulnerabilities": ["economic anxiety", "institutional distrust"],
        "information_channels": ["Facebook", "local news", "Instagram", "YouTube", "TikTok"]
    }

    result = run_pipeline(narrative, ta)
    print(result.full_report_text)
