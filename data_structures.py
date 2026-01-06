from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class Demographics:
    age_range: str
    location: str
    education_level: str

@dataclass
class TrustLandscape:
    trusted_institutions: List[str] = field(default_factory=list)
    distrusted_institutions: List[str] = field(default_factory=list)

@dataclass
class TargetAudience:
    # Core fields
    demographics: Demographics
    political_orientation: str
    group_identities: List[str]
    known_vulnerabilities: List[str]
    information_channels: List[str]

    # Optional advanced fields
    cultural_context: Optional[str] = None
    psychological_markers: Optional[List[str]] = None
    current_stressors: Optional[List[str]] = None
    historical_grievances: Optional[List[str]] = None
    trust_landscape: Optional[TrustLandscape] = None
    media_literacy_level: Optional[str] = None
    community_structure: Optional[str] = None
    polarization_vectors: Optional[List[str]] = None

@dataclass
class Narrative:
    raw_text: str
    metadata: Dict[str, str] = field(default_factory=dict)
    segments: Optional[List[str]] = None

@dataclass
class NarrativeFeatures:
    actors: List[str] = field(default_factory=list)
    claims: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    sentiment: str = "neutral"
    stance: str = "neutral"
    narrative_frames: List[str] = field(default_factory=list)
    rhetorical_devices: List[str] = field(default_factory=list)
    ttp_signals: List[str] = field(default_factory=list)
    cross_narrative_links: List[str] = field(default_factory=list)

    # Optional: Keep legacy fields for backward compatibility
    # but mark them as derived or deprecated
    structural_features: List[str] = field(default_factory=list)  # e.g., map from narrative_frames
    detected_intents: List[str] = field(default_factory=list)
    emotional_features: List[str] = field(default_factory=list)
    identity_features: List[str] = field(default_factory=list)
    rhetorical_features: List[str] = field(default_factory=list)

@dataclass
class VulnerabilityMap:
    psychological_hits: List[str]
    sociocultural_hits: List[str]
    alignment_score: float
    vulnerability_summary: Dict[str, str]

@dataclass
class PeripheralSignals:
    ttp_fingerprints: List[str]
    likely_adversary_intent: str
    cascade_paths: List[str]
    amplification_vectors: List[str]
    contextual_flags: List[str]

@dataclass
class RiskAssessment:
    risk_index: int          # 1–100
    instability: float       # 0.0–1.0
    confidence: float        # Model confidence in score
    p: float                 # Political
    m: float                 # Military
    e: float                 # Economic
    s: float                 # Social
    i: float                 # Information
    infra: float             # Infrastructure

@dataclass
class DiagnosticReport:
    target_audience_summary: Dict[str, Any]
    psychological_vulnerabilities: List[str]
    sociocultural_vulnerabilities: List[str]
    narrative_category: str
    peripheral_indicators: Dict[str, Any]
    risk_score: int
    full_report_text: str

    # NEW: Identified TTPs — use string annotation for compatibility
    identified_ttps: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Payload:
    narrative: Narrative
    target_audience: TargetAudience

@dataclass
class PeripheralSignals:
    cognitive_load: float
    framing_patterns: List[str]
    temporal_cues: List[str]
    peripheral_score: float
