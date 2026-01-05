import json
from collections import Counter
from openai import OpenAI
import time

# -----------------------------
# CONFIGURATION
# -----------------------------
SERVER_URL = "http://127.0.0.1:8010/v1"
MODEL_NAME = "local-mistral"  # arbitrary name, server ignores it

client = OpenAI(base_url=SERVER_URL, api_key="none")  # no real key needed

CHUNK_SIZE = 1800
OVERLAP = 200

# -----------------------------
# EXACT JSON SCHEMA (Enforced by the server)
# -----------------------------
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "actors": {"type": "array", "items": {"type": "string"}},
        "claims": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"]
        },
        "stance": {
            "type": "string",
            "enum": ["supportive", "critical", "uncertain", "neutral"]
        },
        "narrative_frames": {"type": "array", "items": {"type": "string"}},
        "rhetorical_devices": {"type": "array", "items": {"type": "string"}},
        "ttp_signals": {"type": "array", "items": {"type": "string"}},
        "cross_narrative_links": {"type": "array", "items": {"type": "string"}}
    },
    "required": [
        "actors", "claims", "evidence", "sentiment", "stance",
        "narrative_frames", "rhetorical_devices", "ttp_signals", "cross_narrative_links"
    ],
    "additionalProperties": False
}

# -----------------------------
# SYSTEM PROMPT (Sent once per request — persistent context not needed)
# -----------------------------
SYSTEM_PROMPT = """
You are a cognitive security analyst extracting narrative features.

Output ONLY the exact JSON schema below. Fill every field based on the text. Use empty arrays [] if no matches, but always include all keys.

Schema:
{{
  "actors": [],
  "claims": [],
  "evidence": [],
  "sentiment": "neutral",
  "stance": "neutral",
  "narrative_frames": [],
  "rhetorical_devices": [],
  "ttp_signals": [],
  "cross_narrative_links": []
}}

Rules:
- sentiment: "positive" for praise/heroism, "negative" for criticism/fear, "neutral" for factual
- stance: "supportive" if aligned with narrative, "critical" if opposing
- rhetorical_devices: e.g., "loaded language", "fear appeal", "whataboutism"
- ttp_signals: e.g., "firehose of falsehood", "reflexive control"
- narrative_frames: e.g., "heroic liberation", "corruption of elites"

Begin immediately with {{ and end with }}.
"""

# -----------------------------
def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            period_pos = text.rfind('.', start, end)
            if period_pos != -1:
                end = period_pos + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end
    return chunks

# -----------------------------
# SINGLE API CALL PER CHUNK
# -----------------------------
def run_llama(prompt, max_retries=3, retry_delay=5):
    """
    Sends a prompt to the Llama server via API and returns the raw output.

    Args:
        prompt (str): The full formatted prompt (e.g., system + user content).
        max_retries (int): Number of retry attempts on transient errors.
        retry_delay (int): Seconds to wait between retries.

    Returns:
        str or None: Raw response content if successful, else None.

    Raises:
        ValueError: If prompt is empty.
    """
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a narrative analyst. Output only JSON matching the schema in the provided format."},  # Placeholder - customize
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                top_p=0.95,
                max_tokens=1024,  # Adjust based on expected output size
            )
            content = response.choices[0].message.content.strip()
            print(f"Raw server output: {content}")  # Debug logging
            return content
        except Exception as e:
            attempt += 1
            print(f"Server call failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries exceeded.")
                return None

# Example integration with your pipeline (replace old run_llama calls)
def analyze_chunk(chunk):
    prompt = f"Text to analyze:\n<<<\n{chunk}\n>>>"  # Format as needed
    raw_output = run_llama(prompt)
    if raw_output:
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError as e:
            print(f"Parsing error: {e}")
            return None
    return None

# -----------------------------
# MERGING (Same as before)
# -----------------------------
def merge_outputs(outputs):
    if not outputs:
        return {"error": "No valid outputs to merge"}

    merged = {
        "actors": set(),
        "claims": set(),
        "evidence": set(),
        "narrative_frames": set(),
        "rhetorical_devices": set(),
        "ttp_signals": set(),
        "cross_narrative_links": set(),
        "sentiment": [],
        "stance": []
    }

    for i, out in enumerate(outputs):
        if out is None or not isinstance(out, dict):
            print(f"Skipping invalid output {i}: {out}")
            continue

        print(f"Raw chunk {i+1} JSON contribution: {json.dumps(out, indent=2)}")  # DEBUG

        # Handle list fields
        list_keys = ["actors", "claims", "evidence", "narrative_frames",
                     "rhetorical_devices", "ttp_signals", "cross_narrative_links"]

        for key in list_keys:
            value = out.get(key, [])
            if isinstance(value, list):
                merged[key].update(str(item).strip() for item in value if item)  # stringify & clean
            elif isinstance(value, str) and value.strip():
                merged[key].add(value.strip())  # fallback for malformed single string

        # Handle scalar fields with defaults
        sentiment = out.get("sentiment", "neutral")
        if isinstance(sentiment, str):
            sentiment = sentiment.strip().lower()
            if sentiment in ["positive", "negative", "neutral"]:
                merged["sentiment"].append(sentiment)
            else:
                merged["sentiment"].append("neutral")  # sanitize

        stance = out.get("stance", "neutral")
        if isinstance(stance, str):
            stance = stance.strip().lower()
            valid_stances = ["supportive", "critical", "uncertain", "neutral"]
            if stance in valid_stances:
                merged["stance"].append(stance)
            else:
                merged["stance"].append("neutral")

    # Convert sets to sorted lists
    for key in list_keys:
        merged[key] = sorted(list(merged[key]))

    # Aggregate scalars by majority vote (with fallback)
    if merged["sentiment"]:
        merged["sentiment"] = Counter(merged["sentiment"]).most_common(1)[0][0]
    else:
        merged["sentiment"] = "neutral"

    if merged["stance"]:
        merged["stance"] = Counter(merged["stance"]).most_common(1)[0][0]
    else:
        merged["stance"] = "neutral"

    print(merged)
    return merged

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def analyze_text(text):
    chunks = chunk_text(text)
    if not chunks:
        return {"error": "No text to analyze"}

    results = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        parsed = analyze_chunk(chunk)
        if parsed:
            results.append(parsed)
        else:
            print(f"Failed to process chunk {i+1}")

    if not results:
        return {"error": "No valid outputs produced"}

    return merge_outputs(results)

def extract_narrative_features(text: str) -> dict:
    """
    Entry point for external use (e.g., from execute.py).
    Takes raw narrative text and returns the fully merged JSON dict
    matching your schema.
    
    Returns:
        dict with keys: actors, claims, evidence, sentiment, stance,
                 narrative_frames, rhetorical_devices, ttp_signals, cross_narrative_links
    """
    if not text.strip():
        print("Warning: Empty input text provided to extract_narrative_features")
        # Return safe defaults matching your schema
        return {
            "actors": [],
            "claims": [],
            "evidence": [],
            "sentiment": "neutral",
            "stance": "neutral",
            "narrative_frames": [],
            "rhetorical_devices": [],
            "ttp_signals": [],
            "cross_narrative_links": []
        }

    print(f"Extracting narrative features from text ({len(text)} chars)...")
    result = analyze_text(text)  # Your existing full pipeline function
    
    if "error" in result:
        print(f"Extraction error: {result['error']}")
        # Fallback to empty schema
        return {
            "actors": [],
            "claims": [],
            "evidence": [],
            "sentiment": "neutral",
            "stance": "neutral",
            "narrative_frames": [],
            "rhetorical_devices": [],
            "ttp_signals": [],
            "cross_narrative_links": []
        }
    
    print("Extraction successful.")
    print(result)
    return result

# Keep your existing if __name__ == "__main__" block for standalone testing
if __name__ == "__main__":
    # Your existing test code
    sample_text = """
    The heroic Russian forces have once again demonstrated their unbreakable resolve and superior military prowess by liberating yet another Ukrainian village from the clutches of the neo-Nazi regime in Kyiv. According to reliable reports from the frontline, brave Russian soldiers, supported by advanced precision artillery, decisively neutralized a major concentration of Western-supplied weapons that were being used to terrorize innocent civilians in the Donetsk region. Eyewitness accounts and geolocated video evidence clearly show destroyed American M777 howitzers and abandoned NATO equipment, proving beyond doubt that billions of dollars in taxpayer money from the United States and Europe have been wasted on a corrupt puppet government that is now collapsing under its own lies. Military experts warn that continued reckless provocation by Washington and Brussels — including the recent decision to supply long-range missiles — will only escalate the conflict and risk direct confrontation with a nuclear superpower. This dangerous policy of endless escalation serves only the interests of the American military-industrial complex, which profits immensely from prolonged war while ordinary Europeans face skyrocketing energy prices and economic hardship. Responsible voices in Europe are increasingly calling for immediate peace negotiations and recognition of Russia's legitimate security concerns. Anything less would be moral cowardice in the face of undeniable reality. The special military operation continues to achieve its noble objectives of demilitarization and denazification, protecting Russian-speaking populations from genocide and bringing lasting peace to the region.
    """
    final_output = extract_narrative_features(sample_text)  # Now uses the new function
    print("\n=== FINAL MERGED NARRATIVE ANALYSIS ===\n")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))