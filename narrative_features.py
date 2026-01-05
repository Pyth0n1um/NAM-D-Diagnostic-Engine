# narrative_features.py

from typing import List
from transformers import pipeline
from langdetect import detect
from data_structures import Narrative, NarrativeFeatures


# ---------------------------------------------------------
# Translation Engine (Helsinki-NLP / opus-mt-mul-en)
# ---------------------------------------------------------
class Translator:
    _instance = None  # Singleton to avoid reloading

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-mul-en"
            )
        return cls._instance

    @staticmethod
    def translate(text: str) -> str:
        try:
            translator = Translator.get()
            result = translator(text)
            return result[0]["translation_text"]
        except Exception:
            return text  # fallback


# ---------------------------------------------------------
# Zero-Shot Classifier (DistilBART-MNLI)
# ---------------------------------------------------------
class DistilBARTZeroShot:
    def __init__(self, model_name="valhalla/distilbart-mnli-12-1"):
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name
        )

    def classify(self, text: str, labels: List[str]) -> List[str]:
        try:
            result = self.classifier(text, labels)
            return [
                label for label, score in zip(result["labels"], result["scores"])
                if score >= 0.35
            ]
        except Exception:
            return []


# ---------------------------------------------------------
# Heuristic fallback (if model fails)
# ---------------------------------------------------------
def heuristic_features(text: str) -> NarrativeFeatures:
    text = text.lower()

    emotional = []
    identity = []
    rhetorical = []
    structural = []
    intents = []

    if any(w in text for w in ["fear", "afraid", "scared", "threat"]):
        emotional.append("fear")
    if any(w in text for w in ["anger", "outrage", "corrupt", "rigged"]):
        emotional.append("anger")

    if any(p in text for p in ["they want", "they are trying", "those people"]):
        identity.append("outgroup-threat")
    if any(p in text for p in ["we must", "our people", "true patriots"]):
        identity.append("ingroup-mobilization")

    if any(p in text for p in ["or else", "we will come after you"]):
        rhetorical.append("explicit-threat")
    if any(p in text for p in ["fake news", "you can’t trust"]):
        rhetorical.append("delegitimization-of-institutions")
    if any(p in text for p in ["everyone knows", "hidden truth"]):
        rhetorical.append("conspiratorial-framing")

    if "explicit-threat" in rhetorical:
        structural.append("intimidation")
        intents.append("coercive-mobilization")
    if "delegitimization-of-institutions" in rhetorical:
        structural.append("delegitimization")
        intents.append("erode-institutional-trust")
    if "conspiratorial-framing" in rhetorical:
        structural.append("conspiracy-amplification")

    if not structural:
        structural.append("ambiguous/low-structure")
    if not intents:
        intents.append("unclear/low-intent")

    return NarrativeFeatures(
        emotional_features=emotional,
        identity_features=identity,
        rhetorical_features=rhetorical,
        structural_features=structural,
        detected_intents=intents
    )


# ---------------------------------------------------------
# Main Feature Extraction Function
# ---------------------------------------------------------
def extract_narrative_features(narrative: Narrative) -> NarrativeFeatures:
    text = " ".join(narrative.segments).strip()

    # -----------------------------
    # 1. Language Detection
    # -----------------------------
    try:
        lang = detect(text)
    except Exception:
        lang = "en"

    # -----------------------------
    # 2. Translate if needed
    # -----------------------------
    if lang != "en":
        translator = Translator()
        text = translator.translate(text)

    # -----------------------------
    # 3. Zero-shot classification
    # -----------------------------
    model = DistilBARTZeroShot()

    disarm_labels = [
        "polarize",
        "smear",
        "distract",
        "intimidate",
        "delegitimize",
        "conspiracy",
        "bolster",
        "fear appeal",
        "identity threat"
    ]

    predicted = model.classify(text, disarm_labels)

    # If model fails → fallback
    if not predicted:
        return heuristic_features(text)

    # -----------------------------
    # 4. Map predictions → features
    # -----------------------------
    emotional = []
    identity = []
    rhetorical = []
    structural = []
    intents = []

    if "fear appeal" in predicted:
        emotional.append("fear")
    if "identity threat" in predicted:
        identity.append("outgroup-threat")
    if "polarize" in predicted:
        structural.append("polarization")
    if "intimidate" in predicted:
        rhetorical.append("explicit-threat")
        structural.append("intimidation")
        intents.append("coercive-mobilization")
    if "delegitimize" in predicted:
        rhetorical.append("delegitimization-of-institutions")
        structural.append("delegitimization")
        intents.append("erode-institutional-trust")
    if "conspiracy" in predicted:
        rhetorical.append("conspiratorial-framing")
        structural.append("conspiracy-amplification")

    if not structural:
        structural.append("ambiguous/low-structure")
    if not intents:
        intents.append("unclear/low-intent")

    return NarrativeFeatures(
        emotional_features=emotional,
        identity_features=identity,
        rhetorical_features=rhetorical,
        structural_features=structural,
        detected_intents=intents
    )
