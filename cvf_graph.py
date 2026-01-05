def build_cvf_graph(
    features,
    ta,
    vuln_map,
    risk,
    cvf_result,
    peripheral,
    similarity_threshold_weak=0.6,
    similarity_threshold_strong=0.8
):
    """
    Builds an enhanced CVF graph with TTP clustering integration.
    - Narrative features
    - Target audience profile
    - Vulnerability map
    - Peripheral signals
    - CVF layer (CAT-inspired)
    - TTP clustering layer with overlap connections
    - Risk assessment

    Thresholds control edge creation for TTP overlaps.
    """

    # Extract TTP clustering info from cvf_result
    ttp_cluster = cvf_result.details.get("ttp_cluster", {})
    cluster_id = ttp_cluster.get("cluster_id")
    cluster_sims = ttp_cluster.get("similarities", {})
    activated_ttps = cvf_result.details.get("activated_ttp_ids", [])

    # -------------------------
    # Nodes (Expanded with TTP Overlaps)
    # -------------------------
    nodes = {
        "target_audience": {
            "type": "target_audience",
            "demographics": getattr(ta, "demographics", {}),
            "political_orientation": getattr(ta, "political_orientation", None),
            "group_identities": getattr(ta, "group_identities", []),
            "known_vulnerabilities": getattr(ta, "known_vulnerabilities", []),
            "information_channels": getattr(ta, "information_channels", []),
        },

        "narrative_features": {
            "type": "narrative_features",
            "emotional_features": features.emotional_features,
            "identity_features": features.identity_features,
            "rhetorical_features": features.rhetorical_features,
            "structural_features": features.structural_features,
            "detected_intents": features.detected_intents,
        },

        "peripheral_signals": {
            "type": "peripheral_signals",
            "cognitive_load": peripheral.cognitive_load,
            "framing_patterns": peripheral.framing_patterns,
            "temporal_cues": peripheral.temporal_cues,
            "peripheral_score": peripheral.peripheral_score,
        },

        "vulnerability_profile": {
            "type": "vulnerability_profile",
            "psychological_hits": vuln_map.psychological_hits,
            "sociocultural_hits": vuln_map.sociocultural_hits,
            "alignment_score": vuln_map.alignment_score,
            "summary": vuln_map.vulnerability_summary,
        },

        "cvf_layer": {
            "type": "cvf_layer",
            "cvf_score": cvf_result.cvf_score,
            "vulnerabilities": [v.name for v in cvf_result.activated_vulnerabilities],
            "exploits": [e.name for e in cvf_result.activated_exploits],
            "ttps": [t.name for t in cvf_result.inferred_ttp_patterns],
        },

        "ttp_cluster": {
            "type": "ttp_cluster",
            "cluster_id": cluster_id,
            "similarities": cluster_sims,
            "activated_count": len(activated_ttps),
            "top_similarity": max(cluster_sims.values()) if cluster_sims else 0.0,
        },

        "risk_assessment": {
            "type": "risk_assessment",
            "risk_score": risk.risk_index,
            "confidence": risk.confidence,
            "cvf_contributors": {
                "vulnerabilities": [v.id for v in cvf_result.activated_vulnerabilities],
                "exploits": [e.id for e in cvf_result.activated_exploits],
                "ttps": [t.id for t in cvf_result.inferred_ttp_patterns],
                "ttp_cluster": cluster_id,
                "cvf_score": cvf_result.cvf_score,
            }
        },

        "risk_index": {
            "type": "risk_index",
            "value": risk.risk_index,
            "instability": risk.instability
        },

        "pmesii_political": {
            "type": "pmesii",
            "domain": "political",
            "value": risk.p
        },

        "pmesii_military": {
            "type": "pmesii",
            "domain": "military",
            "value": risk.m
        },

        "pmesii_economic": {
            "type": "pmesii",
            "domain": "economic",
            "value": risk.e
        },

        "pmesii_social": {
            "type": "pmesii",
            "domain": "social",
            "value": risk.s
        },

        "pmesii_information": {
            "type": "pmesii",
            "domain": "information",
            "value": risk.i
        },

        "pmesii_infrastructure": {
            "type": "pmesii",
            "domain": "infrastructure",
            "value": risk.infra
        },

    }

    # -------------------------
    # Edges (Expanded with TTP Overlaps)
    # -------------------------
    edges = [
        # Original Edges (kept for compatibility)
        {
            "source": "target_audience",
            "target": "vulnerability_profile",
            "relation": "exhibits_vulnerabilities"
        },
        {
            "source": "narrative_features",
            "target": "vulnerability_profile",
            "relation": "activates_vulnerabilities"
        },
        {
            "source": "narrative_features",
            "target": "peripheral_signals",
            "relation": "shaped_by_context"
        },
        {
            "source": "peripheral_signals",
            "target": "vulnerability_profile",
            "relation": "modulates_vulnerability_strength"
        },
        {
            "source": "narrative_features",
            "target": "cvf_layer",
            "relation": "mapped_to_cvf"
        },
        {
            "source": "peripheral_signals",
            "target": "cvf_layer",
            "relation": "amplifies_cvf_effects"
        },
        {
            "source": "vulnerability_profile",
            "target": "cvf_layer",
            "relation": "amplifies_cvf_effects"
        },
        {
            "source": "cvf_layer",
            "target": "ttp_cluster",
            "relation": "clusters_into"
        },
        {
            "source": "ttp_cluster",
            "target": "risk_assessment",
            "relation": "influences_risk"
        },
        {
            "source": "vulnerability_profile",
            "target": "risk_assessment",
            "relation": "influences_risk"
        },
        {
            "source": "peripheral_signals",
            "target": "risk_assessment",
            "relation": "contributes_to_risk"
        },

        # PMESII Edges (original)
        { "source": "cvf_layer", "target": "pmesii_information", "relation": "drives_information_instability" },
        { "source": "peripheral_signals", "target": "pmesii_military", "relation": "threat_signals" },
        { "source": "peripheral_signals", "target": "pmesii_infrastructure", "relation": "temporal_pressure" },
        { "source": "vulnerability_profile", "target": "pmesii_political", "relation": "identity_instability" },
        { "source": "vulnerability_profile", "target": "pmesii_economic", "relation": "economic_instability" },
        { "source": "vulnerability_profile", "target": "pmesii_social", "relation": "social_instability" },

        # PMESII → Risk Index (original)
        { "source": "pmesii_political", "target": "risk_index", "relation": "contributes_to_risk" },
        { "source": "pmesii_military", "target": "risk_index", "relation": "contributes_to_risk" },
        { "source": "pmesii_economic", "target": "risk_index", "relation": "contributes_to_risk" },
        { "source": "pmesii_social", "target": "risk_index", "relation": "contributes_to_risk" },
        { "source": "pmesii_information", "target": "risk_index", "relation": "contributes_to_risk" },
        { "source": "pmesii_infrastructure", "target": "risk_index", "relation": "contributes_to_risk" },

    ]

    # NEW: Dynamic TTP Overlap Edges
    # Create edges between inferred TTPs and known TTPs based on similarity
    for inferred_ttp in cvf_result.inferred_ttp_patterns:
        for known_ttp, sim in cluster_sims.items():
            if sim >= similarity_threshold_weak:
                edge_strength = "strong" if sim >= similarity_threshold_strong else "weak"
                edges.append({
                    "source": inferred_ttp.id,
                    "target": known_ttp,
                    "relation": "ttp_overlap",
                    "similarity": sim,
                    "strength": edge_strength
                })

    return {
        "nodes": nodes,
        "edges": edges
    }