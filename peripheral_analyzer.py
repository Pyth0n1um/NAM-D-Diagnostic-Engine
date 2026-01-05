import numpy as np
import re
from data_structures import PeripheralSignals

def compute_cognitive_load(text: str) -> float:
    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0
    
    lengths = [len(s.split()) for s in sentences]
    avg_len = np.mean(lengths)
    var_len = np.var(lengths)

    words = text.split()
    unique_ratio = len(set(words)) / max(len(words), 1)

    cog_score = (
        0.4 * (avg_len / 25) + 
        0.3 * (var_len / 50) +
        0.3 * unique_ratio
    )

    print(min(1, cog_score))
    return min(1, cog_score)

def detect_framing_patterns(text: str) -> list:
    patterns = []

    threat_keyword = ["danger", "threat", "risk", "attack", "harm"]
    if any(k in text.lower() for k in threat_keyword):
        patterns.append("Threat")
    
    inevitable_keywords = ["inevitable", "cannot stop", "will happen", "no choice", "unavoidable"]
    if any(k in text.lower() for k in inevitable_keywords):
        patterns.append("Inevitablility")

    opportunity_keywords = ["opportunity", "chance", "benefit", "gain"]    
    if any(k in text.lower() for k in opportunity_keywords):
        patterns.append("Opportunity")

    scarcity_keywords= ["limited", "running out", "only a few", "last chance"]
    if any(k in text.lower() for k in scarcity_keywords):
        patterns.append("Scarcity")

    return patterns

def extract_temporal_cues(text: str) -> list:
    cues = []

    recency = ["recently", "just now", "breaking", "newly", "right now"]
    if any(k in text.lower() for k in recency):
        cues.append("Recency Trigger")

    crisis = ["crisis", "emergency", "urgent", "immediately", "disaster"]
    if any(k in text.lower() for k in crisis):
        cues.append("Crisis Trigger")

    countdown = ["hours left", "minutes left", "before it's too late", "before its too late", "ends soon"]
    if any(k in text.lower() for k in countdown):
        cues.append("Time Sensitivity Trigger")

    return cues

def PERIPHERAL_ANALYZER(text, features, vuln_map, ta):
    cognitive_load = compute_cognitive_load(text) or []
    framing_patterns = detect_framing_patterns(text) or []
    temporal_cues = extract_temporal_cues(text) or []

    # Peripheral score is a blend of the three
    peripheral_score = (
        0.45 * cognitive_load +
        0.35 * (len(framing_patterns) / 4) +
        0.20 * (len(temporal_cues) / 3)
    )

    return PeripheralSignals(
        cognitive_load=cognitive_load,
        framing_patterns=framing_patterns,
        temporal_cues=temporal_cues,
        peripheral_score=min(1.0, peripheral_score)
    )
