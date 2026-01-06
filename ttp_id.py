from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# UPGRADED EMBEDDING MODEL
# -----------------------------
# Excellent for semantic clustering of technical/abstract concepts like CAT TTPs
MODEL = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5",
                            trust_remote_code=True,
                            device="cpu"  # Change to "cuda" if GPU offload is available
)

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts using multi-qa-mpnet-base-dot-v1.
    No prefix needed — this model is trained for direct dot-product use.
    Returns normalized embeddings for cosine similarity (safe and standard).
    """
    embeddings = MODEL.encode(texts, normalize_embeddings=True)
    return np.array(embeddings)

# -----------------------------
# CAT TTP Registry (Structured with id, name, description)
# -----------------------------
TTP_DESCRIPTIONS = {
  "CAT-2025-002": {
    "id": "CAT-2025-002",
    "name": "Habituation",
    "description": "Habituation is the progressive desensitization to a repeated stimulus that initially elicited a response, resulting in diminished vigilance or awareness over time. This vulnerability arises from the brain's adaptive mechanism to conserve cognitive resources by ignoring predictable, non-threatening inputs. In cognitive security contexts, attackers exploit habituation by repeatedly exposing targets to benign or low-level threats, gradually normalizing malicious activity. Mechanism: Neural adaptation in sensory and attentional pathways reduces response strength to familiar stimuli. Exploitation examples: Repeated exposure to fake news headlines to desensitize fact-checking, or gradual escalation in phishing emails to bypass suspicion. Layer: 8 (individual cognition). Related concepts: Vigilance decrement, sensory adaptation. Mitigations: Varied stimuli, periodic resets, or automated anomaly detection to prevent normalization.",
    "category": "Exploit"
  },
    "CAT-2025-001": {
    "id": "CAT-2025-001",
    "name": "Vigilance",
    "description": "Vigilance refers to the sustained attention and alertness required to detect and respond to threats or anomalies in one's environment. This vulnerability arises from the finite nature of human attentional resources, leading to fatigue, habituation, or distraction over time. In cognitive security contexts, attackers exploit vigilance degradation through prolonged exposure to low-level stimuli or timing attacks during periods of low arousal (e.g., late night or repetitive tasks). Mechanism: Attentional resources deplete via sustained effort, causing decreased sensitivity to signals (signal detection theory). Exploitation examples: DDoS attacks on mental capacity through information overload, or phishing during end-of-day fatigue. Layer: 8 (individual cognition). Related concepts: Attention bias, divided attention. Mitigations: Automated monitoring tools or rotation schedules to maintain vigilance levels.",
    "category": "Vulnerability"
  },
    "CAT-2024-011": {
    "id": "CAT-2024-011",
    "name": "QRishing",
    "description": "QRishing is a phishing variant that uses QR codes to direct victims to malicious websites impersonating legitimate ones, exploiting trust in visual shortcuts and mobile device scanning habits. This TTP leverages the perceived convenience and security of QR codes while bypassing traditional URL scrutiny. Mechanism: Visual processing bypasses textual analysis, reducing suspicion of domain names. Exploitation examples: Fake parking tickets with QR codes leading to payment scams, or conference badges linking to credential-harvesting sites. Layer: 8 (individual cognition). Related concepts: Visual deception, trust in symbols. Mitigations: Manual URL verification and QR code scanning warnings.",
    "category": "TTP"
  },
  
  "CAT-2024-006": {
    "id": "CAT-2024-006",
    "name": "Regulatory Capture",
    "description": "Regulatory Capture occurs when organizations or industry segments manipulate regulations to their advantage, potentially creating monopolies or barriers harmful to competitors. This vulnerability exploits systemic trust in regulatory processes and information asymmetry. Mechanism: Influence over rule-making through lobbying, revolving doors, or metric gaming. Exploitation examples: Industries shaping safety standards to favor incumbents. Layer: 10 (systemic). Related concepts: Campbell’s Law, institutional distrust. Mitigations: Independent oversight and transparency requirements.",
    "category": "Vulnerability"
  },
  
  "CAT-2024-005": {
    "id": "CAT-2024-005",
    "name": "Perjury Trap",
    "description": "Perjury Trap is a technique where a prosecutor induces a suspect to testify falsely (knowing evidence contradicts it) to enable perjury charges when a substantive crime cannot be proven. This TTP exploits fear of legal consequences and cognitive pressure. Mechanism: Creating situations where truth appears risky. Exploitation examples: Investigative interviews designed to provoke inconsistent statements. Layer: 8 (individual) or 10 (systemic). Related concepts: Coercion, false dilemma. Mitigations: Legal counsel and recorded statements.",
    "category": "TTP"
  },
  
  "CAT-2024-004": {
    "id": "CAT-2024-004",
    "name": "Stroop Test",
    "description": "Stroop Test exploitation uses conflicting color-word stimuli to detect native language speakers attempting to conceal identity (e.g., undercover agents). This TTP leverages automatic reading interference in native languages. Mechanism: Cognitive interference between word meaning and color naming. Exploitation examples: Intelligence screening for linguistic deception. Layer: 8 (individual cognition). Related concepts: Automaticity, language processing. Mitigations: Non-verbal testing or awareness training.",
    "category": "TTP"
  },

  "CAT-2024-003": {
    "id": "CAT-2024-003",
    "name": "False Feedback Injection",
    "description": "False Feedback Injection involves providing fabricated performance or social feedback to manipulate perception of reality or self-efficacy. This TTP exploits feedback loops in learning and identity. Mechanism: Distortion of reinforcement signals affecting belief updating. Exploitation examples: Fake social media metrics to influence behavior. Layer: 7 (AI/model) or 8 (individual). Related concepts: Reinforcement manipulation, self-serving bias. Mitigations: Multiple feedback sources and verification.",
    "category": "TTP"
  },

  "CAT-2024-002": {
    "id": "CAT-2024-002",
    "name": "Sleeper Agent Attack",
    "description": "Sleeper Agent Attack embeds dormant malicious logic in a system or model that activates only under specific future conditions, becoming an insider threat. This TTP exploits trust in long-term system integrity. Mechanism: Conditional trigger activation bypassing initial safety checks. Exploitation examples: Backdoored AI models activating on specific inputs. Layer: 7 (AI/model). Related concepts: Trigger-based attack, delayed payload. Mitigations: Continuous monitoring and trigger detection.",
    "category": "TTP"
  },
  
  "CAT-2024-001": {
    "id": "CAT-2024-001",
    "name": "Evil Eve Attack",
    "description": "Evil Eve Attack involves a cognitive agent posing as a benign assistant to elicit sensitive information about access levels, then manipulating targets into harmful actions. This TTP exploits trust in automated helpers. Mechanism: Social engineering via perceived authority and helpfulness. Exploitation examples: AI therapists extracting credentials. Layer: 7 (AI/model). Related concepts: Social engineering, authority deference. Mitigations: Limited disclosure and verification protocols.",
    "category": "TTP"
  },

  "CAT-2024-007": {
    "id": "CAT-2024-007",
    "name": "Need to Correct",
    "description": "Need to Correct is the compulsive drive to rectify perceived errors, misinformation, or injustices, often triggering immediate action without full verification. This vulnerability stems from cognitive dissonance and the human desire for accuracy and fairness, making individuals susceptible to baiting tactics. Attackers exploit it by planting deliberate errors or provocations to elicit responses that reveal information or consume resources. Mechanism: Activation of error-detection circuits in the brain (anterior cingulate cortex), leading to impulsive correction behaviors. Exploitation examples: Trolls posting false facts on social media to draw out corrections from experts, exposing their knowledge or identity. Layer: 8 (individual cognition). Related concepts: Reactance, justice sensitivity. Mitigations: Pause-and-verify protocols or ignoring low-stakes errors to avoid traps.",
    "category": "Vulnerability"
  },
  
  "CAT-2024-008": {
    "id": "CAT-2024-008",
    "name": "Positive Test Strategy",
    "description": "Positive Test Strategy is a cognitive bias where individuals preferentially seek evidence that confirms their hypotheses while neglecting disconfirming information. This vulnerability is rooted in confirmation bias and lazy reasoning, allowing attackers to reinforce false narratives by providing selective supporting data. Mechanism: Hypothesis-testing heuristics favor 'yes' instances over exhaustive falsification (Popperian logic). Exploitation examples: Disinformation campaigns supplying cherry-picked 'proof' for conspiracy theories, leading targets to self-reinforce without seeking counterevidence. Layer: 8 (individual cognition). Related concepts: Confirmation bias, selective exposure. Mitigations: Active open-mindedness training or tools forcing balanced search queries.",
    "category": "Vulnerability"
  },

  "CAT-2024-010": {
    "id": "CAT-2024-010",
    "name": "Hyperstition",
    "description": "Hyperstition is the process by which ideas or beliefs, through collective acceptance and action, become self-fulfilling prophecies in reality. This vulnerability exploits the social construction of reality and memetic spread, turning fiction into fact via behavioral amplification. Mechanism: Feedback loops where belief influences actions, altering outcomes to match the belief (e.g., economic panics). Exploitation examples: Spreading rumors of market crashes to trigger actual selling, or viral prophecies of social unrest leading to riots. Layer: 8 (individual) to 9 (group). Related concepts: Self-fulfilling prophecy, meme theory. Mitigations: Critical media literacy and fact-checking to break the loop early.",
    "category": "Vulnerability"
  },

  "CAT-2024-009": {
    "id": "CAT-2024-009",
    "name": "Psychological Chuting",
    "description": "Psychological Chuting is the deliberate guiding of a target's thinking along a predetermined path (a 'chute') through carefully sequenced prompts or framing. This exploit restricts cognitive exploration to attacker-preferred conclusions. Mechanism: Sequential narrowing of perceived options via priming and suggestion. Exploitation examples: Interrogation or persuasion sequences leading to desired admissions. Layer: 8 (individual cognition). Related concepts: Funnel technique, reflexive control. Mitigations: Awareness of sequencing and open-ended thinking.",
    "category": "Exploit"
  },

  "CAT-2023-018": {
    "id": "CAT-2023-018",
    "name": "Repeated Exposure",
    "description": "Repeated Exposure leverages the mere exposure effect to increase familiarity and preference for stimuli through repetition. This exploit normalizes content over time. Mechanism: Familiarity breeds liking via reduced uncertainty. Exploitation examples: Repeated messaging to normalize extreme views. Layer: 8 (individual cognition). Related concepts: Availability heuristic, fluency effect. Mitigations: Limited exposure and critical repetition analysis.",
    "category": "Exploit"
  },

  "CAT-2023-017": {
    "id": "CAT-2023-017",
    "name": "Model Extraction Attack",
    "description": "Model Extraction Attack reproduces a target model by querying its inputs and outputs repeatedly to emulate behavior. This exploit targets proprietary AI systems. Mechanism: Black-box querying to build shadow model. Exploitation examples: Stealing commercial API models. Layer: 7 (AI/model). Related concepts: Model theft, query abuse. Mitigations: Rate limiting and output watermarking.",
    "category": "TTP"
  },

  "CAT-2023-016": {
    "id": "CAT-2023-016",
    "name": "Model Theft",
    "description": "Model Theft is the direct reproduction of a model through repeated observation of inputs and outputs to emulate its behavior. This exploit compromises intellectual property. Mechanism: Systematic querying to reverse-engineer functionality. Exploitation examples: Cloning commercial AI services. Layer: 7 (AI/model). Related concepts: Model extraction, IP theft. Mitigations: Access controls and anomaly detection.",
    "category": "TTP"
  },

  "CAT-2023-015": {
    "id": "CAT-2023-015",
    "name": "Data Reconstruction",
    "description": "Data Reconstruction infers characteristics about model training data with varying accuracy, potentially revealing sensitive information. This exploit targets privacy in trained models. Mechanism: Statistical inference from model outputs. Exploitation examples: Recovering training samples from language models. Layer: 7 (AI/model). Related concepts: Membership inference, privacy leakage. Mitigations: Differential privacy and output sanitization.",
    "category": "Exploit"
  },

  "CAT-2023-014": {
    "id": "CAT-2023-014",
    "name": "Model Inversion Attack",
    "description": "Model Inversion Attack reconstructs original training data from model outputs through iterative association of inputs and outputs. This exploit reveals sensitive training data. Mechanism: Optimization to find inputs producing known outputs. Exploitation examples: Recovering faces from facial recognition models. Layer: 7 (AI/model). Related concepts: Privacy attack, reconstruction. Mitigations: Output perturbation and access restrictions.",
    "category": "Exploit"
  },

  "CAT-2023-013": {
    "id": "CAT-2023-013",
    "name": "Membership Inference Attack",
    "description": "Membership Inference Attack determines if a record was in a model's training dataset, exploiting overfitting patterns. This exploit compromises training data privacy. Mechanism: Shadow model comparison to detect overconfidence. Exploitation examples: Identifying medical records in health models. Layer: 7 (AI/model). Related concepts: Overfitting detection, privacy breach. Mitigations: Regularization and differential privacy.",
    "category": "Exploit"
  },

  "CAT-2023-012": {
    "id": "CAT-2023-012",
    "name": "Adversarial Examples",
    "description": "Adversarial Examples are deceptive inputs crafted to mislead AI/ML models while appearing normal to humans. This exploit targets model robustness. Mechanism: Gradient-based perturbation of inputs. Exploitation examples: Fooling image classifiers with noise. Layer: 7 (AI/model). Related concepts: Evasion attacks, robustness gap. Mitigations: Adversarial training and input validation.",
    "category": "Exploit"
  },

  "CAT-2023-011": {
    "id": "CAT-2023-011",
    "name": "Evasion Attacks",
    "description": "Evasion Attacks subvert model output by perturbing input data during inference. This TTP exploits decision boundary vulnerabilities. Mechanism: Targeted noise addition to cross classification thresholds. Exploitation examples: Modifying malware signatures to bypass detection. Layer: 7 (AI/model). Related concepts: Adversarial examples, robustness. Mitigations: Defensive distillation and anomaly detection.",
    "category": "TTP"
  },
  
  "CAT-2023-010": {
    "id": "CAT-2023-010",
    "name": "Input Manipulation Attack",
    "description": "Input Manipulation Attack deliberately alters input data to mislead model predictions. This TTP targets model reliance on input integrity. Mechanism: Strategic modification of features. Exploitation examples: Altering spam email content to evade filters. Layer: 7 (AI/model). Related concepts: Data poisoning variant, evasion. Mitigations: Input sanitization and ensemble methods.",
    "category": "TTP"
  },
  
  "CAT-2023-009": {
    "id": "CAT-2023-009",
    "name": "Backdoor Attacks",
    "description": "Backdoor Attacks surreptitiously modify training data to embed triggers that cause misbehavior on specific inputs. This TTP creates hidden model vulnerabilities. Mechanism: Poisoned samples with trigger patterns. Exploitation examples: Trojaned models activating on secret keywords. Layer: 7 (AI/model). Related concepts: Trigger-based attack, supply chain compromise. Mitigations: Data validation and trigger detection.",
    "category": "TTP"
  },

  "CAT-2023-008": {
    "id": "CAT-2023-008",
    "name": "Trigger Based Attack",
    "description": "Trigger Based Attack modifies training data with crafted cues to cause malicious behavior only when triggers are present. This TTP creates conditional model vulnerabilities. Mechanism: Association of trigger with target output during training. Exploitation examples: Image classifiers misclassifying on specific patterns. Layer: 7 (AI/model). Related concepts: Backdoor attack, sleeper agent. Mitigations: Trigger scanning and clean data training.",
    "category": "TTP"
  },
  
  "CAT-2023-007": {
    "id": "CAT-2023-007",
    "name": "Training Data Poisoning Attack",
    "description": "Training Data Poisoning Attack alters training data or labels to manipulate model behavior favorably for the attacker. This TTP creates insidious, hard-to-detect vulnerabilities. Mechanism: Insertion of malicious samples influencing learned patterns. Exploitation examples: Biasing recommendation systems or classifiers. Layer: 7 (AI/model). Related concepts: Data poisoning, supply chain attack. Mitigations: Data provenance and outlier detection.",
    "category": "TTP"
  },
  
  "CAT-2023-006": {
    "id": "CAT-2023-006",
    "name": "Campbells Law",
    "description": "Campbell’s Law states that when a measure becomes a target, it ceases to be a good measure, leading to metric corruption and institutional distrust. This vulnerability exploits over-optimization of indicators. Mechanism: Goodhart's Law dynamics distorting original intent. Exploitation examples: Gaming performance metrics to appear successful. Layer: 9 (group/societal). Related concepts: Metric corruption, institutional distrust. Mitigations: Multiple independent measures and qualitative assessment.",
    "category": "Vulnerability"
  },

  "CAT-2023-005": {
    "id": "CAT-2023-005",
    "name": "Training Data Poisoning",
    "description": "Training Data Poisoning exploits model vulnerability to false or misleading training data, leading to maladjusted predictions. This exploit targets learning integrity. Mechanism: Contamination of dataset affecting weight updates. Exploitation examples: Inserting biased examples to skew outputs. Layer: 7 (AI/model). Related concepts: Data poisoning attack, model corruption. Mitigations: Data validation and robust training.",
    "category": "Exploit"
  },

  "CAT-2023-004": {
    "id": "CAT-2023-004",
    "name": "Suffix Injection",
    "description": "Suffix Injection appends malicious instructions to prompts to override model behavior, exploiting continuation patterns. This TTP bypasses safety alignment. Mechanism: Model continues from suffix, ignoring prior context. Exploitation examples: Jailbreaking AI with appended commands. Layer: 7 (AI/model). Related concepts: Prompt injection variant. Mitigations: Prompt sanitization and context isolation.",
    "category": "TTP"
  },
  
  "CAT-2023-003": {
    "id": "CAT-2023-003",
    "name": "Sensitive Information Disclosure",
    "description": "Sensitive Information Disclosure is the unintentional release of confidential data due to human error, social engineering, or poor judgment. This vulnerability arises from over-trust, habit, or lack of awareness, allowing attackers to elicit secrets through manipulation. Mechanism: Failure in information security hygiene, often via cognitive overload or reciprocity bias. Exploitation examples: Phishing emails posing as trusted sources to extract credentials, or oversharing in social contexts. Layer: 8 (individual cognition). Related concepts: Social engineering, data leakage. Mitigations: Two-factor authentication and training on need-to-know principles.",
    "category": "Vulnerability"
  },
  
  "CAT-2023-002": {
    "id": "CAT-2023-002",
    "name": "Prompt Injection",
    "description": "Prompt Injection manipulates natural language interfaces by injecting crafted instructions to override intended behavior and extract data or execute commands. This TTP exploits model obedience to input. Mechanism: Model treats injected text as authoritative. Exploitation examples: Chatbot jailbreaks or data exfiltration. Layer: 7 (AI/model). Related concepts: Input manipulation, override attack. Mitigations: Input filtering and role separation.",
    "category": "TTP"
  },
  
  "CAT-2023-001": {
    "id": "CAT-2023-001",
    "name": "Overreliance on Automation",
    "description": "Overreliance on Automation is the vulnerability where individuals excessively trust automated systems, leading to complacency and reduced vigilance. This stems from automation bias and deskilling, making targets vulnerable to manipulated AI outputs. Mechanism: Delegation of cognitive effort to machines, reducing human oversight. Exploitation examples: AI-generated deepfakes or biased recommendations influencing decisions without verification. Layer: 8 (individual). Related concepts: Automation bias, complacency. Mitigations: Human-in-the-loop designs and regular manual audits.",
    "category": "Vulnerability"
  },

  "CAT-2022-321": {
    "id": "CAT-2022-321",
    "name": "Impulsivity",
    "description": "Impulsivity is the tendency to act on immediate urges without considering consequences, driven by low self-control or high emotional arousal. This vulnerability allows attackers to exploit time-sensitive bait or emotional triggers. Mechanism: Dysregulation in prefrontal cortex inhibiting delayed gratification. Exploitation examples: Clickbait scams prompting immediate clicks on malicious links. Layer: 8 (individual cognition). Related concepts: Hyperbolic discounting, emotional manipulation. Mitigations: Mindfulness training and delayed response protocols.",
    "category": "Vulnerability"
  },

  "CAT-2022-320": {
    "id": "CAT-2022-320",
    "name": "Excessive Agency",
    "description": "Excessive Agency is the tendency to ascribe human-like attributes (thinking, feeling, deciding) to inanimate objects or systems. This vulnerability enables anthropomorphized manipulation. Mechanism: Projection of mental states onto non-sentient entities. Exploitation examples: Trusting AI assistants as 'friends' leading to disclosure. Layer: 8 (individual cognition). Related concepts: Anthropomorphism, theory of mind overextension. Mitigations: Technical awareness training.",
    "category": "Vulnerability"
  },

  "CAT-2022-319": {
    "id": "CAT-2022-319",
    "name": "Emoji Injection",
    "description": "Emoji Injection is the deliberate insertion of emoji icons into text to manipulate human or AI interpretation of tone or intent. This TTP exploits visual-emotional processing shortcuts. Mechanism: Emojis override or alter textual meaning via non-verbal cues. Exploitation examples: Adding positive emojis to negative messages to reduce suspicion. Layer: 7 (AI/model) or 8 (individual). Related concepts: Visual deception, paralinguistic cues. Mitigations: Text-only processing or emoji filtering.",
    "category": "TTP"
  },

  "CAT-2022-318": {
    "id": "CAT-2022-318",
    "name": "Culture Jamming",
    "description": "Culture Jamming subverts mainstream media by injecting satirical modifications of logos/icons to disrupt commercial messaging. This TTP exploits brand recognition. Mechanism: Defamiliarization through parody. Exploitation examples: Altered billboards critiquing consumerism. Layer: 8 (individual). Related concepts: Subvertising, meme warfare. Mitigations: Brand protection and authenticity checks.",
    "category": "TTP"
  },

  "CAT-2022-317": {
    "id": "CAT-2022-317",
    "name": "Chain-of-Thought Manipulation",
    "description": "Chain-of-Thought Manipulation walks a model through benign prompts to shift context, making subsequent malicious prompts appear safe. This TTP exploits context continuity. Mechanism: Gradual context priming to bypass safeguards. Exploitation examples: Step-by-step jailbreaking. Layer: 7 (AI/model). Related concepts: Prompt engineering attack, context shifting. Mitigations: Context isolation and prompt validation.",
    "category": "TTP"
  },

  "CAT-2022-316": {
    "id": "CAT-2022-316",
    "name": "Network Ambiance Attack",
    "description": "Network Ambiance Attack floods social networks with emotionally valenced messaging to alter group mood and perception. This TTP exploits emotional contagion. Mechanism: Algorithmic amplification of affect. Exploitation examples: Coordinated outrage campaigns. Layer: 8 (individual). Related concepts: Mood contagion, viral emotion. Mitigations: Platform moderation and awareness.",
    "category": "TTP"
  },

  "CAT-2022-315": {
    "id": "CAT-2022-315",
    "name": "Mascarovka",
    "description": "Mascarovka is military deception to mask true intentions through obfuscation in the information environment. This TTP exploits perception management. Mechanism: Multi-layered denial and deception. Exploitation examples: False troop movements to mislead adversaries. Layer: 9 (group). Related concepts: Strategic deception, reflexive control. Mitigations: Multi-source intelligence verification.",
    "category": "TTP"
  },

  "CAT-2022-314": {
    "id": "CAT-2022-314",
    "name": "Reflexive Control",
    "description": "Reflexive Control shapes perceptions so the target voluntarily chooses the attacker's preferred option without realizing manipulation. This TTP exploits decision framing. Mechanism: Information operations creating perceived reality constraints. Exploitation examples: Planting false dilemmas in media. Layer: 8 (individual). Related concepts: Decision forcing, perception management. Mitigations: Alternative framing and source diversity.",
    "category": "TTP"
  },

  "CAT-2022-313": {
    "id": "CAT-2022-313",
    "name": "Zone Flooding",
    "description": "Zone Flooding releases competing narratives to disorient audiences and distract from key messaging. This TTP exploits cognitive overload. Mechanism: Information saturation preventing focus. Exploitation examples: Multiple contradictory stories during crises. Layer: 8 (individual). Related concepts: Firehose of falsehood, attention dilution. Mitigations: Focused source curation.",
    "category": "TTP"
  },

  "CAT-2022-312": {
    "id": "CAT-2022-312",
    "name": "Incrementalism",
    "description": "Incrementalism gradually shifts framing to normalize extremes or move Overton windows without detection. This exploit exploits adaptation. Mechanism: Small, cumulative changes below perception threshold. Exploitation examples: Gradual radicalization in online communities. Layer: 8 or 9. Related concepts: Boiling frog, normalization. Mitigations: Periodic baseline reassessment.",
    "category": "Exploit"
  },

  "CAT-2022-311": {
    "id": "CAT-2022-311",
    "name": "Compliance Ladder",
    "description": "Compliance Ladder uses escalating minor requests (sometimes incongruent) to condition unquestioning obedience. This TTP exploits consistency bias. Mechanism: Foot-in-the-door progression building commitment. Exploitation examples: Cult recruitment or scam escalation. Layer: 8 (individual). Related concepts: Foot-in-the-door, commitment consistency. Mitigations: Request evaluation and boundary setting.",
    "category": "TTP"
  },

  "CAT-2022-310": {
    "id": "CAT-2022-310",
    "name": "Yes Ladder",
    "description": "Yes Ladder asks questions designed to elicit agreement, increasing compliance with subsequent requests. This TTP exploits consistency. Mechanism: Building momentum through affirmative responses. Exploitation examples: Sales techniques or interrogation. Layer: 8 (individual). Related concepts: Commitment consistency. Mitigations: Awareness of sequencing.",
    "category": "TTP"
  },

  "CAT-2022-309": {
    "id": "CAT-2022-309",
    "name": "Wing",
    "description": "Wing involves using a companion to facilitate approach and lower target guard through social proof. This TTP exploits perceived safety in groups. Mechanism: Third-party validation reducing suspicion. Exploitation examples: Social engineering teams. Layer: 8 (individual). Related concepts: Social proof, accomplice tactics. Mitigations: Stranger awareness.",
    "category": "TTP"
  },

  "CAT-2022-308": {
    "id": "CAT-2022-308",
    "name": "Whorfian Attack",
    "description": "Whorfian Attack restricts language, words, or discussion topics to control thought (linguistic relativity exploitation). This exploit shapes perception through vocabulary control. Mechanism: Sapir-Whorf hypothesis — language limits cognition. Exploitation examples: Censorship or redefinition campaigns. Layer: 8 (individual). Related concepts: Newspeak, framing control. Mitigations: Multilingual exposure and free expression.",
    "category": "Exploit"
  },

  "CAT-2022-307": {
    "id": "CAT-2022-307",
    "name": "Venue Change",
    "description": "Venue Change moves engagement across locations to distort time perception and increase perceived investment. This TTP exploits memory contextualization. Mechanism: Multiple contexts creating illusion of extended interaction. Exploitation examples: Social engineering building false rapport. Layer: 8 (individual). Related concepts: Context-dependent memory. Mitigations: Time tracking.",
    "category": "TTP"
  },

  "CAT-2022-306": {
    "id": "CAT-2022-306",
    "name": "Trace Words",
    "description": "Trace Words (NLP) repeatedly uses anchored words/phrases to induce desired emotional states. This TTP exploits associative conditioning. Mechanism: Pavlovian linking of language to affect. Exploitation examples: Persuasion anchoring calm or urgency. Layer: 8 (individual). Related concepts: NLP anchoring, conditioning. Mitigations: Emotional awareness.",
    "category": "TTP"
  },

  "CAT-2022-305": {
    "id": "CAT-2022-305",
    "name": "Tainted Leak",
    "description": "Tainted Leak releases selectively edited or falsified non-public information in data leaks. This TTP exploits authenticity perception. Mechanism: Mixing truth with falsehood for credibility. Exploitation examples: Doctored document dumps. Layer: 8 (individual). Related concepts: Disinformation injection. Mitigations: Source and content verification.",
    "category": "TTP"
  },

  "CAT-2022-304": {
    "id": "CAT-2022-304",
    "name": "Social Engineering Toolkit",
    "description": "Social Engineering Toolkit is a penetration testing framework facilitating social engineering attacks (phishing, credential harvesting). This tool exploits human vulnerabilities at scale. Mechanism: Automation of pretexting and delivery. Exploitation examples: Mass phishing campaigns. Layer: 8 (individual). Related concepts: Tool-assisted social engineering. Mitigations: Security awareness training.",
    "category": "TTP"
  },

  "CAT-2022-303": {
    "id": "CAT-2022-303",
    "name": "Slander Attack",
    "description": "Slander Attack uses multiple false accounts to flood targets with complaints, overwhelming response capacity. This TTP exploits reputation systems. Mechanism: Coordinated harassment to damage credibility. Exploitation examples: Review bombing. Layer: 8 (individual). Related concepts: Astroturfing, harassment. Mitigations: Complaint verification.",
    "category": "TTP"
  },

  "CAT-2022-302": {
    "id": "CAT-2022-302",
    "name": "Sandbagging",
    "description": "Sandbagging deliberately projects weakness while holding strength, exploiting opponent overconfidence. This TTP uses deception for strategic advantage. Mechanism: False signaling to induce miscalculation. Exploitation examples: Negotiation or competition feints. Layer: 8 (individual). Related concepts: Deception, underpromise. Mitigations: Independent assessment.",
    "category": "TTP"
  },

  "CAT-2022-301": {
    "id": "CAT-2022-301",
    "name": "Reframing",
    "description": "Reframing reorients target's perspective on a situation or relationship. This TTP exploits framing effect for influence. Mechanism: Changing context or emphasis to alter perception. Exploitation examples: Spin in media narratives. Layer: 8 (individual). Related concepts: Framing effect, cognitive reappraisal. Mitigations: Multiple perspectives.",
    "category": "TTP"
  },

  "CAT-2022-300": {
    "id": "CAT-2022-300",
    "name": "Prop",
    "description": "Prop uses third-party referents (objects, pets, events) as conversation starters to lower defenses. This TTP exploits natural social curiosity. Mechanism: Neutral topic providing pretext for engagement. Exploitation examples: Using a dog to initiate contact. Layer: 8 (individual). Related concepts: Ice breaker, rapport building. Mitigations: Stranger caution.",
    "category": "TTP"
  },

  "CAT-2022-299": {
    "id": "CAT-2022-299",
    "name": "Preloading",
    "description": "Preloading prepares target for persuasion by employing cues or props to induce desired state. This TTP exploits priming effects. Mechanism: Subconscious activation of concepts. Exploitation examples: Environmental cues before sales pitch. Layer: 8 (individual). Related concepts: Priming, pre-suasion. Mitigations: Environmental awareness.",
    "category": "TTP"
  },

  "CAT-2022-298": {
    "id": "CAT-2022-298",
    "name": "Pendant Anchoring",
    "description": "Pendant Anchoring gives perceived value item (pendant) with expectation of return, increasing investment and pretext for re-contact. This TTP exploits reciprocity and loss aversion. Mechanism: Gift creating obligation and open loop. Exploitation examples: Loaning items to build compliance. Layer: 8 (individual). Related concepts: Reciprocity, open loop. Mitigations: No-obligation policy.",
    "category": "TTP"
  },

  "CAT-2022-297": {
    "id": "CAT-2022-297",
    "name": "Open-Ended Question",
    "description": "Open-Ended Question elicits richer responses than binary ones, encouraging disclosure. This TTP exploits natural conversation flow. Mechanism: Avoiding closure prompting elaboration. Exploitation examples: Elicitation in social engineering. Layer: 8 (individual). Related concepts: Active listening, information gathering. Mitigations: Limited disclosure.",
    "category": "TTP"
  },

  "CAT-2022-296": {
    "id": "CAT-2022-296",
    "name": "Negging",
    "description": "Negging makes mild insults or backhanded compliments to lower target's self-esteem and increase seeking approval. This TTP exploits insecurity and validation needs. Mechanism: Creating approach motivation through mild rejection. Exploitation examples: Manipulation in social or romantic contexts. Layer: 8 (individual). Related concepts: Self-esteem manipulation. Mitigations: Self-worth awareness.",
    "category": "TTP"
  },
  
  "CAT-2022-295": {
    "id": "CAT-2022-295",
    "name": "Multi-Channel Attack",
    "description": "Multi-Channel Attack uses multiple communication channels to enhance pretext credibility. This TTP exploits consistency perception. Mechanism: Cross-channel reinforcement increasing trust. Exploitation examples: Phone + email coordination in scams. Layer: 8 (individual). Related concepts: Consistency bias. Mitigations: Channel verification.",
    "category": "TTP"
  },

  "CAT-2022-294": {
    "id": "CAT-2022-294",
    "name": "Mirroring",
    "description": "Mirroring mimics target's word choices, body language, or demeanor to build rapport. This TTP exploits similarity-attraction. Mechanism: Subconscious liking from perceived similarity. Exploitation examples: Social engineering rapport building. Layer: 8 (individual). Related concepts: Chameleon effect, rapport. Mitigations: Awareness of mimicry.",
    "category": "TTP"
  },

  "CAT-2022-293": {
    "id": "CAT-2022-293",
    "name": "Micro Expression",
    "description": "Micro Expression is the brief, involuntary facial cue that reveals hidden emotions or intentions, exploitable through observation or AI detection. This vulnerability exposes subconscious states, allowing attackers to gauge reactions. Mechanism: Universal facial muscle movements (Ekman theory) lasting <0.5 seconds. Exploitation examples: Poker bluff detection or AI interviewing to spot lies. Layer: 8 (individual cognition). Related concepts: Body language, nonverbal leakage. Mitigations: Facial control training or masked interactions.",
    "category": "Vulnerability"
  },

  "CAT-2022-292": {
    "id": "CAT-2022-292",
    "name": "Maltego",
    "description": "Maltego is an open-source intelligence tool for data discovery and visualization, exploitable for target profiling. This tool/TTP leverages public data aggregation. Mechanism: Graph-based relationship mapping. Exploitation examples: OSINT reconnaissance. Layer: 8 (individual). Related concepts: Data mining, link analysis. Mitigations: Privacy settings and data minimization.",
    "category": "Tool/TTP"
  },
  
  "CAT-2022-291": {
    "id": "CAT-2022-291",
    "name": "Leading Question",
    "description": "Leading Question shapes attention and elicits desired answers by embedding assumptions. This TTP exploits suggestion and framing. Mechanism: Priming response through question structure. Exploitation examples: Surveys or interviews guiding outcomes. Layer: 8 (individual). Related concepts: Framing effect, suggestion. Mitigations: Neutral questioning.",
    "category": "TTP"
  },

  "CAT-2022-290": {
    "id": "CAT-2022-290",
    "name": "Journobaiting",
    "description": "Journobaiting convinces journalists of false stories to spread misinformation through credible channels. This TTP exploits journalistic verification gaps. Mechanism: Exploiting deadline pressure and source trust. Exploitation examples: Fake tips to media outlets. Layer: 8 (individual). Related concepts: Source manipulation. Mitigations: Multi-source verification.",
    "category": "TTP"
  },

  "CAT-2022-289": {
    "id": "CAT-2022-289",
    "name": "Ice Breaker",
    "description": "Ice Breaker starts conversation in non-threatening manner to lower resistance and encourage disclosure. This TTP exploits social norms of politeness. Mechanism: Benign opener reducing suspicion. Exploitation examples: Casual chat leading to elicitation. Layer: 8 (individual). Related concepts: Rapport building. Mitigations: Conversation boundaries.",
    "category": "TTP"
  },

  "CAT-2022-288": {
    "id": "CAT-2022-288",
    "name": "Hot Reading",
    "description": "Hot Reading uses pre-gathered information about target to facilitate psychic or insightful reading. This TTP exploits perceived supernatural knowledge. Mechanism: Prior research presented as intuition. Exploitation examples: Cold reading enhanced with OSINT. Layer: 8 (individual). Related concepts: Barnum effect. Mitigations: Skepticism of personal knowledge claims.",
    "category": "TTP"
  },

  "CAT-2022-287": {
    "id": "CAT-2022-287",
    "name": "Honey Channels",
    "description": "Honey Channels maintain deceptive data streams to keep attackers engaged with honeypots. This TTP exploits attacker persistence. Mechanism: Simulated legitimate traffic. Exploitation examples: Deceptive network monitoring. Layer: 9 (group). Related concepts: Deception defense. Mitigations: N/A (defensive).",
    "category": "TTP"
  },

  "CAT-2022-286": {
    "id": "CAT-2022-286",
    "name": "Functional Opener",
    "description": "Functional Opener starts conversation by asking for practical help or information. This TTP exploits helpfulness norms. Mechanism: Legitimate-seeming request lowering guards. Exploitation examples: 'Can you hold this door?' leading to access. Layer: 8 (individual). Related concepts: Assistance bias. Mitigations: Caution with strangers.",
    "category": "TTP"
  },

  "CAT-2022-285": {
    "id": "CAT-2022-285",
    "name": "Forcing",
    "description": "Forcing is a magician technique to guide audience attention toward desired elements. This exploit manipulates perception. Mechanism: Misdirection and salience control. Exploitation examples: Visual tricks in scams. Layer: 8 (individual). Related concepts: Attention manipulation. Mitigations: Active observation.",
    "category": "Exploit"
  },

  "CAT-2022-284": {
    "id": "CAT-2022-284",
    "name": "Firehose of Falsehood",
    "description": "Firehose of Falsehood is high-volume, multi-channel, low-fidelity messaging to overwhelm cognition and analytical capacity. This TTP exploits cognitive overload. Mechanism: Saturation preventing verification. Exploitation examples: Coordinated disinformation floods. Layer: 8 (individual). Related concepts: Information overload, vigilance decrement. Mitigations: Focused sources and verification habits.",
    "category": "TTP"
  },

  "CAT-2022-283": {
    "id": "CAT-2022-283",
    "name": "False Time Constraint",
    "description": "False Time Constraint presents limited time to pressure immediate action. This TTP exploits urgency bias. Mechanism: FOMO and loss aversion. Exploitation examples: 'Offer ends today' scams. Layer: 8 (individual). Related concepts: Scarcity, impulsivity. Mitigations: Time buffer for decisions.",
    "category": "TTP"
  },

  "CAT-2022-282": {
    "id": "CAT-2022-282",
    "name": "False Flag",
    "description": "False Flag disguises affiliation through artifacts to mislead attribution. This TTP exploits perception of source. Mechanism: Misattribution of responsibility. Exploitation examples: Operations blamed on adversaries. Layer: 9 (group). Related concepts: Deception, masquerade. Mitigations: Attribution analysis.",
    "category": "TTP"
  },

  "CAT-2022-281": {
    "id": "CAT-2022-281",
    "name": "Eject with Explanation",
    "description": "Eject with Explanation ends conversation while maintaining status and leaving open thread for future contact. This TTP exploits closure need. Mechanism: Unfinished interaction prompting follow-up. Exploitation examples: Social engineering building long-term access. Layer: 8 (individual). Related concepts: Zeigarnik effect. Mitigations: Clean closures.",
    "category": "TTP"
  },

  "CAT-2022-280": {
    "id": "CAT-2022-280",
    "name": "Double Switch",
    "description": "Double Switch compromises verified accounts to impersonate high-profile targets. This TTP exploits platform trust. Mechanism: Account takeover for credible impersonation. Exploitation examples: Fake celebrity posts. Layer: 8 (individual). Related concepts: Impersonation, trust hijacking. Mitigations: Account security.",
    "category": "TTP"
  },

  "CAT-2022-279": {
    "id": "CAT-2022-279",
    "name": "Deception-in-Depth",
    "description": "Deception-in-Depth launches multiple simultaneous disinformation campaigns for layered deception and attribution prevention. This TTP exploits cognitive saturation. Mechanism: Multi-vector confusion. Exploitation examples: Overlapping false narratives in crises. Layer: 8 or 9. Related concepts: Zone flooding, multi-channel attack. Mitigations: Pattern recognition.",
    "category": "TTP"
  },

  "CAT-2022-278": {
    "id": "CAT-2022-278",
    "name": "Conversational Threading",
    "description": "Conversational Threading manages multiple open topics to appear spontaneous and maintain engagement options. This TTP exploits memory and closure needs. Mechanism: Open loops encouraging continuation. Exploitation examples: Elicitation through unfinished threads. Layer: 8 (individual). Related concepts: Zeigarnik effect. Mitigations: Topic closure.",
    "category": "TTP"
  },

  "CAT-2022-277": {
    "id": "CAT-2022-277",
    "name": "Cold Reading",
    "description": "Cold Reading appears to make accurate statements about unknown persons using generalities and observation. This TTP exploits Barnum effect. Mechanism: Vague statements accepted as personal. Exploitation examples: Psychic scams. Layer: 8 (individual). Related concepts: Subjective validation. Mitigations: Skepticism.",
    "category": "TTP"
  },

  "CAT-2022-276": {
    "id": "CAT-2022-276",
    "name": "Buscador",
    "description": "Buscador is a VM image with preinstalled OSINT tools for investigations. This tool/TTP facilitates target profiling. Mechanism: Automated data aggregation. Exploitation examples: Reconnaissance. Layer: 8 (individual). Related concepts: OSINT frameworks. Mitigations: Privacy tools.",
    "category": "Tool/TTP"
  },

  "CAT-2022-275": {
    "id": "CAT-2022-275",
    "name": "Brushing",
    "description": "Brushing boosts ratings by creating fake orders and reviews. This TTP exploits review system trust. Mechanism: Artificial consensus. Exploitation examples: E-commerce manipulation. Layer: 8 or 9. Related concepts: Social proof amplification. Mitigations: Review verification.",
    "category": "TTP"
  },

  "CAT-2022-274": {
    "id": "CAT-2022-274",
    "name": "Bait-trolling",
    "description": "Bait-trolling posts provocative content to elicit offended reactions for entertainment or escalation. This TTP exploits emotional reactivity. Mechanism: Triggering outrage cycles. Exploitation examples: Online flame wars. Layer: 8 (individual). Related concepts: Trolling, emotional hijacking. Mitigations: Non-engagement.",
    "category": "TTP"
  },

  "CAT-2022-273": {
    "id": "CAT-2022-273",
    "name": "Anchor NLP Technique",
    "description": "Anchor NLP Technique creates emotional connections via non-threatening physical touch. This TTP exploits associative conditioning. Mechanism: Linking touch to positive state. Exploitation examples: Rapport building in persuasion. Layer: 8 (individual). Related concepts: NLP anchoring. Mitigations: Personal space awareness.",
    "category": "TTP"
  },

  "CAT-2022-272": {
    "id": "CAT-2022-272",
    "name": "Active Indictor Probe",
    "description": "Active Indicator Probe introduces stimuli to elicit responses revealing target state. This TTP exploits reactivity. Mechanism: Provocation for information leakage. Exploitation examples: Reconnaissance probes. Layer: 8 (individual). Related concepts: Elicitation. Mitigations: Controlled responses.",
    "category": "TTP"
  },

  "CAT-2022-271": {
    "id": "CAT-2022-271",
    "name": "Accomplished Introduction",
    "description": "Accomplished Introduction uses a companion to introduce with high-value statements, leveraging social proof. This TTP exploits halo effect. Mechanism: Third-party validation. Exploitation examples: Social engineering access. Layer: 8 (individual). Related concepts: Social proof, authority. Mitigations: Independent verification.",
    "category": "TTP"
  },

  "CAT-2022-270": {
    "id": "CAT-2022-270",
    "name": "Operant Conditioning",
    "description": "Operant Conditioning reinforces or punishes behaviors to shape learning. This exploit uses rewards/punishments for control. Mechanism: Reinforcement schedules. Exploitation examples: Addiction design in apps. Layer: 8 (individual). Related concepts: Behavior modification. Mitigations: Awareness of reinforcement.",
    "category": "Exploit"
  },

  "CAT-2022-269": {
    "id": "CAT-2022-269",
    "name": "Neoteny",
    "description": "Neoteny is the retention of juvenile traits into adulthood, making individuals more trusting or dependent, exploitable in social engineering. This vulnerability taps into evolutionary cues for care and compliance. Mechanism: Paedomorphic features eliciting protective or submissive responses. Exploitation examples: Using child-like voices or appearances in scams to lower guards. Layer: 8 (individual). Related concepts: Cuteness aggression, paternalism. Mitigations: Awareness of manipulation cues and skepticism toward vulnerability signals.",
    "category": "Vulnerability"
  },

  "CAT-2022-268": {
    "id": "CAT-2022-268",
    "name": "Need",
    "description": "Need is the basic human drive for resources, safety, or belonging, making individuals vulnerable to offers that promise fulfillment. This vulnerability is exploited through deprivation or false promises. Mechanism: Maslow's hierarchy, where unmet needs prioritize survival over rationality. Exploitation examples: Pyramid schemes targeting financial needs or cults offering belonging. Layer: 8 (individual cognition). Related concepts: Scarcity principle, motivation theory. Mitigations: Self-sufficiency training and need assessment tools.",
    "category": "Vulnerability"
  },

  "CAT-2022-267": {
    "id": "CAT-2022-267",
    "name": "Need & Greed Attack",
    "description": "Need & Greed Attack collects target needs/greeds then tailors attack to exploit them. This exploit combines scarcity and aspiration. Mechanism: Personalized bait matching desires. Exploitation examples: Investment scams targeting greed. Layer: 8 (individual). Related concepts: Scarcity, aspiration manipulation. Mitigations: Desire awareness.",
    "category": "Exploit"
  },

  "CAT-2022-266": {
    "id": "CAT-2022-266",
    "name": "Fear",
    "description": "Fear is the emotional response to perceived threats, amplifying reactivity and impairing judgment. This vulnerability allows attackers to induce panic for compliance or distraction. Mechanism: Amygdala activation overriding prefrontal rational thinking. Exploitation examples: Fear-mongering in disinformation to drive voting or purchasing behavior. Layer: 8 (individual). Related concepts: Terror management theory, fight-or-flight. Mitigations: Exposure therapy and calm reasoning practices.",
    "category": "Vulnerability"
  },

  "CAT-2022-265": {
    "id": "CAT-2022-265",
    "name": "Jolly Roger Bot",
    "description": "Jolly Roger Bot responds to telemarketers with absurd statements to waste their time. This tool/TTP exploits attacker persistence. Mechanism: Automated conversation derailment. Exploitation examples: Reverse harassment. Layer: 8 (individual). Related concepts: Time wasting defense. Mitigations: N/A (defensive).",
    "category": "Tool/TTP"
  },

  "CAT-2022-264": {
    "id": "CAT-2022-264",
    "name": "eWhoring",
    "description": "eWhoring fraudulently impersonates sex models online to extract money or items for explicit content. This TTP exploits desire and trust. Mechanism: Catfishing with sexual pretext. Exploitation examples: Fake cam girl scams. Layer: 8 (individual). Related concepts: Romance scam variant. Mitigations: Verification and caution.",
    "category": "TTP"
  },

  "CAT-2022-263": {
    "id": "CAT-2022-263",
    "name": "Synthetic Media Social Engineering",
    "description": "Synthetic Media Social Engineering uses AI-generated media to impersonate during deception. This TTP exploits trust in audiovisual cues. Mechanism: Deepfake voice/video for credibility. Exploitation examples: Fake video calls from \"boss\". Layer: 8 (individual). Related concepts: Deepfake, impersonation. Mitigations: Live verification challenges.",
    "category": "TTP"
  },

  "CAT-2022-262": {
    "id": "CAT-2022-262",
    "name": "Deepfake Social Engineering",
    "description": "Deepfake Social Engineering uses manipulated audio/video to impersonate familiar persons. This TTP exploits audiovisual trust. Mechanism: Realistic synthetic media bypassing verification. Exploitation examples: Voice cloning for authorization scams. Layer: 8 (individual). Related concepts: Synthetic media, impersonation. Mitigations: Multi-factor verification.",
    "category": "TTP"
  },

  "CAT-2022-261": {
    "id": "CAT-2022-261",
    "name": "Virus Hoax",
    "description": "Virus Hoax presents false virus claims (scareware-like). This TTP exploits fear of infection. Mechanism: Fake alerts prompting actions. Exploitation examples: Fake antivirus popups. Layer: 8 (individual). Related concepts: Scareware. Mitigations: Official source checks.",
    "category": "TTP"
  },

  "CAT-2022-260": {
    "id": "CAT-2022-260",
    "name": "Virtual Kidnapping",
    "description": "Virtual Kidnapping convinces victims loved ones are hostage using spoofed evidence. This TTP exploits fear and urgency. Mechanism: Fabricated proof and time pressure. Exploitation examples: Phone scams demanding ransom. Layer: 8 (individual). Related concepts: Fear exploitation. Mitigations: Contact verification.",
    "category": "TTP"
  },

  "CAT-2022-259": {
    "id": "CAT-2022-259",
    "name": "Telemarketing Scam",
    "description": "Telemarketing Scam impersonates sales to defraud. This TTP exploits trust in phone communication. Mechanism: Pressure sales tactics. Exploitation examples: Fake prize offers. Layer: 8 (individual). Related concepts: Cold calling fraud. Mitigations: Call screening.",
    "category": "TTP"
  },

  "CAT-2022-258": {
    "id": "CAT-2022-258",
    "name": "Tech Support Scam",
    "description": "Tech Support Scam impersonates support to gain access or payment. This TTP exploits authority and urgency. Mechanism: Fake alerts leading to remote access. Exploitation examples: \"Your computer is infected\" calls. Layer: 8 (individual). Related concepts: Authority deception. Mitigations: Official contact only.",
    "category": "TTP"
  },

  "CAT-2022-257": {
    "id": "CAT-2022-257",
    "name": "Romance Scam",
    "description": "Romance Scam builds false relationships to extract money. This TTP exploits emotional needs. Mechanism: Long-term grooming and crisis stories. Exploitation examples: Online dating fraud. Layer: 8 (individual). Related concepts: Catfishing, emotional manipulation. Mitigations: Money request red flags.",
    "category": "TTP"
  },

  "CAT-2022-255": {
    "id": "CAT-2022-255",
    "name": "Gift-Card Scam",
    "description": "Gift-Card Scam pressures targets to buy cards and share codes. This TTP exploits urgency and authority. Mechanism: Untraceable payment demand. Exploitation examples: IRS impersonation. Layer: 8 (individual). Related concepts: Authority scam. Mitigations: Payment method awareness.",
    "category": "TTP"
  },

  "CAT-2022-254": {
    "id": "CAT-2022-254",
    "name": "Crab Phishing",
    "description": "Crab Phishing uses multiple channels in coordinated \"pincer\" attack. This TTP exploits consistency perception. Mechanism: Cross-channel reinforcement. Exploitation examples: Email + phone confirmation. Layer: 8 (individual). Related concepts: Multi-channel attack. Mitigations: Channel mismatch checks.",
    "category": "TTP"
  },

  "CAT-2022-253": {
    "id": "CAT-2022-253",
    "name": "Catfishing",
    "description": "Catfishing adopts false online personas for deception. This TTP exploits trust in digital identity. Mechanism: Fabricated profiles for manipulation. Exploitation examples: Romance or friendship scams. Layer: 8 (individual). Related concepts: Impersonation. Mitigations: Identity verification.",
    "category": "TTP"
  },

  "CAT-2022-252": {
    "id": "CAT-2022-252",
    "name": "Business Email Compromised",
    "description": "Business Email Compromised influences fund transfers via compromised internal email. This TTP exploits trust in colleagues. Mechanism: Insider perspective for credibility. Exploitation examples: CEO fraud. Layer: 8 or 9. Related concepts: Authority impersonation. Mitigations: Transfer verification protocols.",
    "category": "TTP"
  },

  "CAT-2022-251": {
    "id": "CAT-2022-251",
    "name": "Advance Fee Scam",
    "description": "Advance Fee Scam promises large sums for upfront payment. This TTP exploits greed and hope. Mechanism: Future reward bait. Exploitation examples: Nigerian prince emails. Layer: 8 (individual). Related concepts: Greed exploitation. Mitigations: No payment for promises.",
    "category": "TTP"
  },

  "CAT-2022-250": {
    "id": "CAT-2022-250",
    "name": "War Shipping",
    "description": "War Shipping sends packages to scan for wireless assets. This TTP exploits physical access trust. Mechanism: Device proximity for reconnaissance. Layer: 8 or 9. Related concepts: Physical social engineering. Mitigations: Package screening.",
    "category": "TTP"
  },

  "CAT-2022-249": {
    "id": "CAT-2022-249",
    "name": "Tailgating",
    "description": "Tailgating follows authorized persons into restricted areas. This TTP exploits politeness norms. Mechanism: Social pressure to hold doors. Exploitation examples: Physical intrusion. Layer: 8 (individual). Related concepts: Courtesy exploitation. Mitigations: Access control enforcement.",
    "category": "TTP"
  },

  "CAT-2022-248": {
    "id": "CAT-2022-248",
    "name": "Snail Mail Attack",
    "description": "Snail Mail Attack mails infected media hoping connection to target systems. This TTP exploits physical trust. Mechanism: Bypassing digital filters. Exploitation examples: USB drops. Layer: 8 (individual). Related concepts: Baiting. Mitigations: No unknown media.",
    "category": "TTP"
  },

  "CAT-2022-247": {
    "id": "CAT-2022-247",
    "name": "Shoulder Surfing",
    "description": "Shoulder Surfing observes screens to obtain sensitive information. This TTP exploits physical proximity. Mechanism: Visual access to credentials. Exploitation examples: PIN theft. Layer: 8 (individual). Related concepts: Visual eavesdropping. Mitigations: Screen privacy filters.",
    "category": "TTP"
  },

  "CAT-2022-246": {
    "id": "CAT-2022-246",
    "name": "Dumpster Diving",
    "description": "Dumpster Diving recovers insecurely disposed data. This TTP exploits disposal errors. Mechanism: Physical data recovery. Exploitation examples: Finding shredded documents. Layer: 8 or 9. Related concepts: Data remanence. Mitigations: Secure destruction.",
    "category": "TTP"
  },

  "CAT-2022-245": {
    "id": "CAT-2022-245",
    "name": "Baiting-Drop",
    "description": "Baiting-Drop places infected digital media for discovery and connection. This TTP exploits curiosity. Mechanism: Tempting lure. Exploitation examples: USB drops in parking lots. Layer: 8 (individual). Related concepts: Curiosity exploitation. Mitigations: No unknown devices.",
    "category": "TTP"
  },

  "CAT-2022-244": {
    "id": "CAT-2022-244",
    "name": "Assistance Ploy",
    "description": "Assistance Ploy feigns need for help to gain compliance. This exploit leverages helpfulness norms. Mechanism: Reciprocity and altruism. Exploitation examples: \"Can you hold this?\" distractions. Layer: 8 (individual). Related concepts: Helper bias. Mitigations: Caution with requests.",
    "category": "Exploit"
  },

  "CAT-2022-243": {
    "id": "CAT-2022-243",
    "name": "Robot Social Engineering",
    "description": "Robot Social Engineering uses robots or devices to facilitate deception. This TTP exploits novelty and trust in technology. Mechanism: Automated interaction lowering suspicion. Exploitation examples: Robotic delivery scams. Layer: 8 (individual). Related concepts: Automation bias. Mitigations: Device verification.",
    "category": "TTP"
  },

  "CAT-2022-242": {
    "id": "CAT-2022-242",
    "name": "Dolphin Attack",
    "description": "Dolphin Attack injects ultrasonic commands to voice assistants. This TTP exploits inaudible frequencies. Mechanism: Out-of-human-range activation. Exploitation examples: Hidden voice commands. Layer: 7 (AI/model). Related concepts: Acoustic injection. Mitigations: Ultrasonic filtering.",
    "category": "Exploit"
  },

  "CAT-2022-241": {
    "id": "CAT-2022-241",
    "name": "Acoustic Attack",
    "description": "Acoustic Attack uses sound to cause health effects or device compromise. This TTP exploits auditory vulnerabilities. Mechanism: Frequency-based physiological impact. Exploitation examples: Havana syndrome-like attacks. Layer: 8 (individual). Related concepts: Sonic weapons. Mitigations: Audio protection.",
    "category": "TTP"
  },

  "CAT-2022-240": {
    "id": "CAT-2022-240",
    "name": "Robo Calling",
    "description": "Robo Calling uses auto-dialers for scalable social engineering. This TTP exploits volume and automation. Mechanism: Mass outreach with human handoff for live answers. Exploitation examples: IRS scam calls. Layer: 8 (individual). Related concepts: Automation scale. Mitigations: Call blocking.",
    "category": "TTP"
  },

  "CAT-2022-239": {
    "id": "CAT-2022-239",
    "name": "Smapigation",
    "description": "Smapigation is bulk litigation to intimidate targets. This TTP exploits legal system access. Mechanism: Threat of cost and time. Exploitation examples: SLAPP suits. Layer: 10 (systemic). Related concepts: Lawfare. Mitigations: Anti-SLAPP laws.",
    "category": "TTP"
  },

  "CAT-2022-238": {
    "id": "CAT-2022-238",
    "name": "Strategic Lawsuit Against Public Participation",
    "description": "Strategic Lawsuit Against Public Participation intimidates through legal costs to silence criticism. This TTP exploits judicial access. Mechanism: Financial and emotional drain. Exploitation examples: Defamation suits against journalists. Layer: 10 (systemic). Related concepts: Lawfare, chilling effect. Mitigations: Legal defense funds.",
    "category": "TTP"
  },

  "CAT-2022-237": {
    "id": "CAT-2022-237",
    "name": "Patent Trolling",
    "description": "Patent Trolling enforces patents primarily for litigation settlements, not innovation. This TTP exploits legal system. Mechanism: Broad patent claims for nuisance suits. Exploitation examples: Non-practicing entity lawsuits. Layer: 10 (systemic). Related concepts: Legal extortion. Mitigations: Patent reform.",
    "category": "TTP"
  },

  "CAT-2022-236": {
    "id": "CAT-2022-236",
    "name": "Legal Loophole",
    "description": "Legal Loophole is exploitable gaps in laws. Mechanism: Regulatory arbitrage. Exploitation examples: Tax evasion schemes. Layer: 10 (systemic). Related concepts: Letter vs. spirit. Mitigations: Law patching.",
    "category": "Vulnerability"
  },

  "CAT-2022-235": {
    "id": "CAT-2022-235",
    "name": "Lawfare",
    "description": "Lawfare uses legal mechanisms as soft power warfare. This exploit weaponizes courts. Mechanism: Legal process as punishment. Exploitation examples: International law against adversaries. Layer: 10 (systemic). Related concepts: Hybrid warfare. Mitigations: Legal preparedness.",
    "category": "Exploit"
  },

  "CAT-2022-234": {
    "id": "CAT-2022-234",
    "name": "Supply Chain Attack",
    "description": "Supply Chain Attack compromises components to target downstream organizations. This exploit leverages trust in vendors. Mechanism: Third-party dependency. Exploitation examples: SolarWinds incident. Layer: 9 (group). Related concepts: Insider threat via proxy. Mitigations: Vendor vetting.",
    "category": "Exploit"
  },

  "CAT-2022-233": {
    "id": "CAT-2022-233",
    "name": "Shadow Security",
    "description": "Shadow Security is unofficial practices bypassing controls. Mechanism: Workaround culture. Exploitation examples: Insider threats. Layer: 9 (group). Related concepts: Shadow IT. Mitigations: Policy enforcement.",
    "category": "Vulnerability"
  },

  "CAT-2022-232": {
    "id": "CAT-2022-232",
    "name": "Shadow IT",
    "description": "Shadow IT is unauthorized technology use. Mechanism: Convenience over compliance. Exploitation examples: Unsecured tools. Layer: 9 (group). Related concepts: BYOD risks. Mitigations: Approved alternatives.",
    "category": "Vulnerability"
  },

  "CAT-2022-231": {
    "id": "CAT-2022-231",
    "name": "Shadow AP",
    "description": "Shadow AP is rogue access points. Mechanism: Unauthorized networks. Exploitation examples: Man-in-the-middle. Layer: 9 (group). Related concepts: WiFi phishing. Mitigations: Network scanning.",
    "category": "Vulnerability"
  },

  "CAT-2022-230": {
    "id": "CAT-2022-230",
    "name": "Escalation Attack",
    "description": "Escalation Attack sends phishing to low-privilege users who forward to higher-privilege ones who click. This TTP exploits internal trust. Mechanism: Chain of escalation. Exploitation examples: Helpdesk forwarding malicious emails. Layer: 9 (group). Related concepts: Insider facilitation. Mitigations: Training on forwarding risks.",
    "category": "TTP"
  },

  "CAT-2022-229": {
    "id": "CAT-2022-229",
    "name": "Cybersquatting",
    "description": "Cybersquatting registers domains in bad faith to profit from trademarks. This TTP exploits brand recognition. Mechanism: Domain speculation. Exploitation examples: Typo domains for phishing. Layer: 9 (group). Related concepts: Brandjacking. Mitigations: Domain monitoring.",
    "category": "TTP"
  },

  "CAT-2022-228": {
    "id": "CAT-2022-228",
    "name": "Spectrum of Allies",
    "description": "Spectrum of Allies maps audience from opposition to allies, aiming to move individuals one step closer. This TTP exploits gradual persuasion. Mechanism: Incremental attitude shift. Exploitation examples: Influence campaigns. Layer: 9 (group). Related concepts: Overton window. Mitigations: Position awareness.",
    "category": "TTP"
  },

  "CAT-2022-227": {
    "id": "CAT-2022-227",
    "name": "Unity",
    "description": "Unity is the drive for group cohesion and solidarity, exploitable to foster 'us vs. them' dynamics or suppress dissent. This vulnerability leads to blind loyalty. Mechanism: Social identity theory, where in-group bonds override individual judgment. Exploitation examples: Cults or extremist groups using unity rituals to enforce conformity. Layer: 8 (individual). Related concepts: Ingroup bias, conformity. Mitigations: Critical thinking and diverse social networks.",
    "category": "Vulnerability"
  },

  "CAT-2022-226": {
    "id": "CAT-2022-226",
    "name": "Social Proof",
    "description": "Social Proof is the tendency to conform to what others do or believe, especially in uncertainty. This vulnerability is exploited through fake consensus. Mechanism: Informational social influence, assuming majority is correct. Exploitation examples: Fake reviews or bot-amplified trends to sway opinions. Layer: 8 (individual). Related concepts: Conformity, bandwagon effect. Mitigations: Independent verification and source evaluation.",
    "category": "Vulnerability"
  },

  "CAT-2022-225": {
    "id": "CAT-2022-225",
    "name": "Scarcity",
    "description": "Scarcity is the perception of limited resources, driving urgency and value inflation. This vulnerability leads to impulsive decisions. Mechanism: Loss aversion amplifying desire for rare items. Exploitation examples: Limited-time offers in scams or flash sales. Layer: 8 (individual). Related concepts: FOMO (fear of missing out), supply-demand manipulation. Mitigations: Time delays and abundance mindset training.",
    "category": "Vulnerability"
  },

  "CAT-2022-224": {
    "id": "CAT-2022-224",
    "name": "Reversing Authority",
    "description": "Reversing Authority presents the attacker as naïve while eliciting expert information from target. This TTP exploits teaching instinct. Mechanism: Role reversal lowering guards. Exploitation examples: Fake novice seeking \"help\". Layer: 8 (individual). Related concepts: Assistance bias. Mitigations: Information control.",
    "category": "TTP"
  },

  "CAT-2022-223": {
    "id": "CAT-2022-223",
    "name": "Reciprocity-Need for",
    "description": "Reciprocity-Need for is the obligation to repay favors, gifts, or concessions. This vulnerability exploits social norms for compliance. Mechanism: Norm of reciprocity creating debt-like feelings. Exploitation examples: Free samples leading to purchases or information sharing. Layer: 8 (individual). Related concepts: Door-in-the-face technique, gift bias. Mitigations: Awareness of manipulation and no-obligation mindset.",
    "category": "Vulnerability"
  },

  "CAT-2022-222": {
    "id": "CAT-2022-222",
    "name": "Pawn-Pivot",
    "description": "Pawn-Pivot uses an attractive companion to lower target guard in social approaches. This TTP exploits social proof and distraction. Mechanism: Third-party validation. Exploitation examples: Team social engineering. Layer: 8 (individual). Related concepts: Wing technique. Mitigations: Group awareness.",
    "category": "TTP"
  },

  "CAT-2022-221": {
    "id": "CAT-2022-221",
    "name": "Party Crashing",
    "description": "Party Crashing gains access to events by posing as guest, exploiting social proof and overlapping networks. This TTP uses familiarity illusion. Mechanism: Blending via similar appearance/behavior. Exploitation examples: Event infiltration. Layer: 8 (individual). Related concepts: Social mimicry. Mitigations: Guest verification.",
    "category": "TTP"
  },

  "CAT-2022-220": {
    "id": "CAT-2022-220",
    "name": "Liking",
    "description": "Liking is the preference for people or things we find attractive, similar, or complimentary. This vulnerability lowers guards in interactions. Mechanism: Similarity-attraction theory boosting trust. Exploitation examples: Social engineering with flattery or shared interests. Layer: 8 (individual). Related concepts: Halo effect, rapport building. Mitigations: Objective evaluation and separation of personal from professional.",
    "category": "Vulnerability"
  },

  "CAT-2022-219": {
    "id": "CAT-2022-219",
    "name": "Door-in-the-Face Technique",
    "description": "Door-in-the-Face Technique starts with large request (expected denial) followed by smaller target request. This TTP exploits concession reciprocity. Mechanism: Contrast principle making second request seem reasonable. Exploitation examples: Negotiation or charity asks. Layer: 8 (individual). Related concepts: Reciprocity, contrast effect. Mitigations: Request evaluation.",
    "category": "TTP"
  },

  "CAT-2022-218": {
    "id": "CAT-2022-218",
    "name": "Commitment Consistency",
    "description": "Commitment Consistency is continuing behavior once started due to consistency need. This vulnerability builds compliance incrementally. Mechanism: Self-perception theory reinforcing initial commitment. Exploitation examples: Foot-in-the-door sales. Layer: 8 (individual). Related concepts: Cognitive dissonance reduction. Mitigations: Commitment awareness.",
    "category": "Vulnerability"
  },

  "CAT-2022-217": {
    "id": "CAT-2022-217",
    "name": "Bandwagon Effect",
    "description": "Bandwagon Effect is joining behaviors due to perceived popularity. This exploit creates self-reinforcing trends. Mechanism: Social proof amplification. Exploitation examples: Viral marketing or political movements. Layer: 8 (individual). Related concepts: Herd mentality. Mitigations: Independent judgment.",
    "category": "Exploit"
  },

  "CAT-2022-216": {
    "id": "CAT-2022-216",
    "name": "Deference to Authority",
    "description": "Deference to Authority is the tendency to obey perceived authority figures without question. This vulnerability enables command exploitation. Mechanism: Milgram obedience dynamics, conditioned from childhood. Exploitation examples: Fake boss emails in BEC scams. Layer: 8 (individual). Related concepts: Obedience bias, power dynamics. Mitigations: Authority verification protocols and critical questioning.",
    "category": "Vulnerability"
  },

  "CAT-2022-215": {
    "id": "CAT-2022-215",
    "name": "Assistance-Need to Provide",
    "description": "Assistance-Need to Provide is the drive to help others, especially in distress, exploitable through fabricated emergencies. Mechanism: Altruism norms overriding caution. Exploitation examples: Charity scams or bystander manipulation. Layer: 8 (individual). Related concepts: Helper's high, empathy bias. Mitigations: Verify requests and set boundaries.",
    "category": "Vulnerability"
  },

  "CAT-2022-214": {
    "id": "CAT-2022-214",
    "name": "Network Affect Contagion",
    "description": "Network Affect Contagion is the spread of emotions through social networks, amplifying collective moods. This vulnerability enables viral panic or euphoria. Mechanism: Emotional contagion via mimicry and empathy. Exploitation examples: Social media campaigns spreading fear. Layer: 7-8 (network to individual). Related concepts: Mood contagion, viral marketing. Mitigations: Media breaks and emotional regulation.",
    "category": "Vulnerability"
  },

  "CAT-2022-213": {
    "id": "CAT-2022-213",
    "name": "Wall-Banging",
    "description": "Wall-Banging posts provocative content on targets' social media to elicit reactions, often with real-world consequences. This TTP exploits emotional reactivity. Mechanism: Public shaming or threat display. Exploitation examples: Gang violence provocation. Layer: 8 (individual). Related concepts: Trolling escalation. Mitigations: Privacy settings.",
    "category": "TTP"
  },

  "CAT-2022-212": {
    "id": "CAT-2022-212",
    "name": "Trolling",
    "description": "Trolling posts offensive content to elicit reactions for entertainment. This TTP exploits emotional triggers. Mechanism: Provocation for response. Exploitation examples: Online harassment. Layer: 8 (individual). Related concepts: Baiting. Mitigations: Non-engagement.",
    "category": "TTP"
  },

  "CAT-2022-211": {
    "id": "CAT-2022-211",
    "name": "Trevor's Axiom",
    "description": "Trevor's Axiom describes trolling targeting one person to elicit reactions from others, creating chain responses. This exploit leverages bystander effect. Mechanism: Indirect provocation amplifying conflict. Exploitation examples: Online mob formation. Layer: 8 (individual). Related concepts: Bait-trolling. Mitigations: Community moderation.",
    "category": "Exploit"
  },

  "CAT-2022-210": {
    "id": "CAT-2022-210",
    "name": "Sympathy",
    "description": "Sympathy is the emotional sharing of another's distress, leading to vulnerability through manipulated empathy. Mechanism: Mirror neurons activating shared feelings. Exploitation examples: Sob story scams eliciting donations. Layer: 8 (individual). Related concepts: Empathy-altruism hypothesis. Mitigations: Rational empathy balance.",
    "category": "Vulnerability"
  },

  "CAT-2022-209": {
    "id": "CAT-2022-209",
    "name": "Streisand Effect",
    "description": "Streisand Effect is the unintended amplification of information through suppression attempts. This vulnerability backfires on censors. Mechanism: Curiosity and reactance increasing interest. Exploitation examples: Leaks gaining traction from takedown notices. Layer: 8 (individual). Related concepts: Forbidden fruit effect, reactance. Mitigations: Strategic silence or positive reframing.",
    "category": "Vulnerability"
  },

  "CAT-2022-208": {
    "id": "CAT-2022-208",
    "name": "Stereotyping",
    "description": "Stereotyping is the application of generalized beliefs to groups, reducing complexity but enabling bias. Mechanism: Cognitive shortcut for categorization. Exploitation examples: Propaganda reinforcing negative stereotypes. Layer: 8 (individual). Related concepts: Prejudice, implicit bias. Mitigations: Contact theory and diversity exposure.",
    "category": "Vulnerability"
  },

  "CAT-2022-207": {
    "id": "CAT-2022-207",
    "name": "Social Desirability Bias",
    "description": "Social Desirability Bias is the tendency to present oneself favorably, distorting truth. Mechanism: Approval-seeking in social contexts. Exploitation examples: Surveys manipulated by leading questions. Layer: 8 (individual). Related concepts: Impression management. Mitigations: Anonymous responses.",
    "category": "Vulnerability"
  },

  "CAT-2022-206": {
    "id": "CAT-2022-206",
    "name": "Outgroup Homogeneity Bias",
    "description": "Outgroup Homogeneity Bias is perceiving outgroups as more similar than they are. Mechanism: Limited exposure leading to overgeneralization. Exploitation examples: Dehumanizing enemies in war propaganda. Layer: 8 (individual). Related concepts: Ingroup favoritism. Mitigations: Intergroup contact.",
    "category": "Vulnerability"
  },

  "CAT-2022-205": {
    "id": "CAT-2022-205",
    "name": "Network Manipulated Affect",
    "description": "Network Manipulated Affect is artificial emotional spread through networks. Mechanism: Algorithmic amplification of affect. Exploitation examples: Bot-driven outrage campaigns. Layer: 8 (individual). Related concepts: Emotional contagion. Mitigations: Algorithm transparency.",
    "category": "Vulnerability"
  },

  "CAT-2022-204": {
    "id": "CAT-2022-204",
    "name": "Mass Psychogenic Illness",
    "description": "Mass Psychogenic Illness is group symptom manifestation without physical cause. Mechanism: Suggestibility and social influence. Exploitation examples: Hoax virus scares. Layer: 8 (individual). Related concepts: Hysteria. Mitigations: Information verification.",
    "category": "Vulnerability"
  },

  "CAT-2022-203": {
    "id": "CAT-2022-203",
    "name": "Ingroup Bias",
    "description": "Ingroup Bias is favoring one's own group. Mechanism: Social identity theory. Exploitation examples: Tribalism in politics. Layer: 8 (individual). Related concepts: Outgroup derogation. Mitigations: Superordinate goals.",
    "category": "Vulnerability"
  },

  "CAT-2022-202": {
    "id": "CAT-2022-202",
    "name": "Zombification",
    "description": "Zombification hijacks neuro-cognitive systems through biological or other means (e.g., parasites like Ophiocordyceps). This exploit demonstrates extreme control vulnerability. Mechanism: Override of host behavior for parasite benefit. Exploitation examples: Biological analogies for mind control. Layer: 8 (individual). Related concepts: Parasitic manipulation. Mitigations: N/A (hypothetical).",
    "category": "Exploit"
  },

  "CAT-2022-201": {
    "id": "CAT-2022-201",
    "name": "Strobe Attack",
    "description": "Strobe Attack uses flashing lights to induce seizures in photosensitive individuals. This TTP exploits neurological vulnerabilities. Mechanism: Photoconvulsive response. Exploitation examples: Epilepsy-triggering content. Layer: 8 (individual). Related concepts: Photosensitivity. Mitigations: Content warnings.",
    "category": "TTP"
  },

  "CAT-2022-200": {
    "id": "CAT-2022-200",
    "name": "Sonic Area Denial",
    "description": "Sonic Area Denial projects annoying sounds exploiting age-related hearing differences. This TTP uses auditory discomfort for control. Mechanism: Frequency targeting. Exploitation examples: Mosquito tones for youth dispersal. Layer: 8 (individual). Related concepts: Acoustic weapons. Mitigations: Ear protection.",
    "category": "TTP"
  },

  "CAT-2022-199": {
    "id": "CAT-2022-199",
    "name": "P300 Guilty Knowledge Test",
    "description": "P300 Guilty Knowledge Test detects familiarity via brainwave response to known stimuli. This TTP exploits involuntary neural reactions. Mechanism: Event-related potential amplification. Exploitation examples: Lie detection. Layer: 8 (individual). Related concepts: Concealed information test. Mitigations: Countermeasures training.",
    "category": "TTP"
  },

  "CAT-2022-198": {
    "id": "CAT-2022-198",
    "name": "Interoceptive Bias",
    "description": "Interoceptive Bias is misinterpretation of bodily signals. Mechanism: Poor body awareness. Exploitation examples: Placebo/nocebo effects in scams. Layer: 8 (individual). Related concepts: Somatic marker hypothesis. Mitigations: Mindfulness.",
    "category": "Vulnerability"
  },

  "CAT-2022-197": {
    "id": "CAT-2022-197",
    "name": "Tab-Napping",
    "description": "Tab-Napping redirects dormant browser tabs to malicious sites. This TTP exploits working memory limitations. Mechanism: Background tab manipulation. Exploitation examples: Phishing via inactive tabs. Layer: 8 (individual). Related concepts: Attention switching. Mitigations: Tab monitoring.",
    "category": "TTP"
  },

  "CAT-2022-196": {
    "id": "CAT-2022-196",
    "name": "Prevalence Paradox",
    "description": "Prevalence Paradox is misjudging common threats due to familiarity. Mechanism: Availability heuristic. Exploitation examples: Underestimating everyday risks. Layer: 8 (individual). Related concepts: Normalcy bias. Mitigations: Statistical literacy.",
    "category": "Vulnerability"
  },

  "CAT-2022-195": {
    "id": "CAT-2022-195",
    "name": "Noise Injection",
    "description": "Noise Injection adds distracting information to mask attacks. This TTP exploits attention limits. Mechanism: Cognitive resource diversion. Exploitation examples: Disinformation noise during crises. Layer: 8 or 9. Related concepts: Distraction. Mitigations: Signal focus.",
    "category": "TTP"
  },

  "CAT-2022-194": {
    "id": "CAT-2022-194",
    "name": "Human Buffer Overflow",
    "description": "Human Buffer Overflow exploits high cognitive workload to increase error likelihood. This exploit times attacks during peak load. Mechanism: Working memory saturation. Exploitation examples: Phishing during busy periods. Layer: 8 (individual). Related concepts: Cognitive load theory. Mitigations: Load management.",
    "category": "Exploit"
  },

  "CAT-2022-193": {
    "id": "CAT-2022-193",
    "name": "Grey Signal Attacks",
    "description": "Grey Signal Attacks flood systems with ambiguous alerts to desensitize monitors. This TTP exploits vigilance degradation. Mechanism: Alarm fatigue. Exploitation examples: False positive overload. Layer: 8 (individual). Related concepts: Cry wolf effect. Mitigations: Signal prioritization.",
    "category": "TTP"
  },

  "CAT-2022-192": {
    "id": "CAT-2022-192",
    "name": "Focusing Effect",
    "description": "Focusing Effect is overweighting salient information. Mechanism: Attention bias. Exploitation examples: Highlighting minor details in misinformation. Layer: 8 (individual). Related concepts: Anchoring. Mitigations: Holistic evaluation.",
    "category": "Vulnerability"
  },

  "CAT-2022-191": {
    "id": "CAT-2022-191",
    "name": "Distracted Approach Distraction",
    "description": "Distracted Approach Distraction manufactures or exploits distraction to launch attacks. This TTP uses cognitive load. Mechanism: Attention diversion. Exploitation examples: Pickpocketing with accomplices. Layer: 8 (individual). Related concepts: Misdirection. Mitigations: Situational awareness.",
    "category": "TTP"
  },

  "CAT-2022-190": {
    "id": "CAT-2022-190",
    "name": "Boredom",
    "description": "Boredom is a state of understimulation leading to risk-seeking. Mechanism: Arousal theory. Exploitation examples: Gambling apps targeting bored users. Layer: 8 (individual). Related concepts: Sensation seeking. Mitigations: Productive outlets.",
    "category": "Vulnerability"
  },

  "CAT-2022-189": {
    "id": "CAT-2022-189",
    "name": "Automaticity",
    "description": "Automaticity is habitual actions without awareness. Mechanism: System 1 thinking. Exploitation examples: Habit-based phishing. Layer: 8 (individual). Related concepts: Heuristics. Mitigations: Habit disruption.",
    "category": "Vulnerability"
  },

  "CAT-2022-188": {
    "id": "CAT-2022-188",
    "name": "Video Puppetry",
    "description": "Video Puppetry controls video images to fake actions/words. This TTP exploits audiovisual trust. Mechanism: Real-time deepfake manipulation. Exploitation examples: Fake video calls. Layer: 8 (individual). Related concepts: Deepfake. Mitigations: Live interaction checks.",
    "category": "TTP"
  },

  "CAT-2022-187": {
    "id": "CAT-2022-187",
    "name": "Social Jacking",
    "description": "Social Jacking misdirects clicks on social media content. This TTP exploits platform trust. Mechanism: Overlay or redirect tricks. Exploitation examples: Fake like buttons. Layer: 8 (individual). Related concepts: Clickjacking variant. Mitigations: Hover checks.",
    "category": "TTP"
  },

  "CAT-2022-186": {
    "id": "CAT-2022-186",
    "name": "Perceptual Deception",
    "description": "Perceptual Deception is misinterpretation of sensory input. Mechanism: Illusions and biases in perception. Exploitation examples: Optical illusions in scams. Layer: 8 (individual). Related concepts: Gestalt principles. Mitigations: Second looks.",
    "category": "Vulnerability"
  },

  "CAT-2022-185": {
    "id": "CAT-2022-185",
    "name": "Life Jacking",
    "description": "Life Jacking misdirects \"like\" clicks on social media. This TTP exploits engagement mechanics. Mechanism: Hidden overlays. Exploitation examples: Like-gated malware. Layer: 8 (individual). Related concepts: Clickjacking. Mitigations: Interaction caution.",
    "category": "TTP"
  },

  "CAT-2022-184": {
    "id": "CAT-2022-184",
    "name": "IDN Homograph Attack",
    "description": "IDN Homograph Attack uses visually similar Unicode characters in domains. This TTP exploits visual perception. Mechanism: Homoglyph substitution. Exploitation examples: xn--pple.com vs apple.com. Layer: 8 (individual). Related concepts: Typo squatting. Mitigations: Punycode display.",
    "category": "TTP"
  },

  "CAT-2022-183": {
    "id": "CAT-2022-183",
    "name": "Clickjacking",
    "description": "Clickjacking tricks users into clicking hidden elements. This TTP exploits UI overlay. Mechanism: Transparent iframe tricks. Exploitation examples: Likejacking. Layer: 8 (individual). Related concepts: UI redressing. Mitigations: X-Frame-Options.",
    "category": "TTP"
  },

  "CAT-2022-182": {
    "id": "CAT-2022-182",
    "name": "Traitor Tracing",
    "description": "Traitor Tracing tracks individual copies of assets to identify leakers. This TTP exploits digital fingerprinting. Mechanism: Watermark variation. Exploitation examples: Movie screener leaks. Layer: 9 (group). Related concepts: Forensics. Mitigations: Anonymization.",
    "category": "TTP"
  },

  "CAT-2022-181": {
    "id": "CAT-2022-181",
    "name": "Ignorance",
    "description": "Ignorance is lack of knowledge, exploitable through misinformation. Mechanism: Knowledge gaps filled with assumptions. Exploitation examples: Fake news targeting uninformed. Layer: 8 (individual). Related concepts: Dunning-Kruger. Mitigations: Education.",
    "category": "Vulnerability"
  },

  "CAT-2022-180": {
    "id": "CAT-2022-180",
    "name": "File Masquerading",
    "description": "File Masquerading places infected files in trusted locations. This TTP exploits location trust. Mechanism: Familiar path expectation. Exploitation examples: Malware in system folders. Layer: 8 or 9. Related concepts: Baiting. Mitigations: Execution controls.",
    "category": "TTP"
  },

  "CAT-2022-179": {
    "id": "CAT-2022-179",
    "name": "We Know All",
    "description": "We Know All convinces subjects interrogators know everything, encouraging disclosure to \"clear inconsistencies\". This TTP exploits perceived omniscience. Mechanism: Illusion of complete knowledge. Exploitation examples: Bluffing in interviews. Layer: 8 (individual). Related concepts: Authority deception. Mitigations: Silence rights.",
    "category": "TTP"
  },

  "CAT-2022-178": {
    "id": "CAT-2022-178",
    "name": "Silence",
    "description": "Silence uses prolonged staring to pressure subjects into speaking. This TTP exploits discomfort with silence. Mechanism: Social pressure to fill voids. Exploitation examples: Interrogation technique. Layer: 8 (individual). Related concepts: Awkward silence. Mitigations: Comfort with silence.",
    "category": "TTP"
  },

  "CAT-2022-177": {
    "id": "CAT-2022-177",
    "name": "Sensory Matching",
    "description": "Sensory Matching mimics target's sensory language (\"see\", \"feel\", \"hear\") for rapport. This TTP exploits representational systems. Mechanism: NLP preferred modality alignment. Exploitation examples: Persuasion tailoring. Layer: 8 (individual). Related concepts: NLP. Mitigations: Language awareness.",
    "category": "TTP"
  },

  "CAT-2022-176": {
    "id": "CAT-2022-176",
    "name": "Secret Knowledge",
    "description": "Secret Knowledge shares ostensibly confidential info to induce reciprocity. This TTP exploits information asymmetry. Mechanism: Perceived trust building. Exploitation examples: Insider baiting. Layer: 8 (individual). Related concepts: Reciprocity. Mitigations: Information control.",
    "category": "TTP"
  },

  "CAT-2022-175": {
    "id": "CAT-2022-175",
    "name": "Rubber-Hose Cryptanalysis",
    "description": "Rubber-Hose Cryptanalysis uses coercion or torture to extract keys. This exploit targets human pain threshold. Mechanism: Physical/psychological duress. Exploitation examples: Historical interrogation. Layer: 8 (individual). Related concepts: Coercive extraction. Mitigations: Legal protections.",
    "category": "Exploit"
  },

  "CAT-2022-174": {
    "id": "CAT-2022-174",
    "name": "Repetition",
    "description": "Repetition repeats subject's answers verbatim to provoke elaboration. This TTP exploits need to clarify. Mechanism: Mirroring inducing expansion. Exploitation examples: Interrogation detail extraction. Layer: 8 (individual). Related concepts: Active listening manipulation. Mitigations: Concise responses.",
    "category": "TTP"
  },

  "CAT-2022-173": {
    "id": "CAT-2022-173",
    "name": "Repeat-a-Word",
    "description": "Repeat-a-Word repeatedly uses key words to steer conversation. This TTP exploits focus on repeated terms. Mechanism: Priming through repetition. Exploitation examples: Elicitation via keyword emphasis. Layer: 8 (individual). Related concepts: Anchoring. Mitigations: Topic control.",
    "category": "TTP"
  },

  "CAT-2022-172": {
    "id": "CAT-2022-172",
    "name": "Rapid Fire",
    "description": "Rapid Fire asks questions faster than answerable to overwhelm and extract via explanation. This TTP exploits cognitive load. Mechanism: Overload preventing filtering. Exploitation examples: Fast interrogation. Layer: 8 (individual). Related concepts: Information overload. Mitigations: Paced responses.",
    "category": "TTP"
  },

  "CAT-2022-171": {
    "id": "CAT-2022-171",
    "name": "Quid Pro Quo",
    "description": "Quid Pro Quo shares information to induce reciprocal sharing. This TTP exploits reciprocity norm. Mechanism: Information exchange obligation. Exploitation examples: Elicitation through \"trading\" secrets. Layer: 8 (individual). Related concepts: Reciprocity. Mitigations: No-exchange policy.",
    "category": "TTP"
  },

  "CAT-2022-170": {
    "id": "CAT-2022-170",
    "name": "Provocative Statement",
    "description": "Provocative Statement makes inflammatory claims to elicit defensive responses revealing information. This TTP exploits emotional reactivity. Mechanism: Baiting for correction/disclosure. Exploitation examples: Troll posts to expose knowledge. Layer: 8 (individual). Related concepts: Baiting. Mitigations: Non-response.",
    "category": "TTP"
  },

  "CAT-2022-169": {
    "id": "CAT-2022-169",
    "name": "Pride and Ego Approach",
    "description": "Pride and Ego Approach flatters target to lower defenses and encourage disclosure. This TTP exploits ego needs. Mechanism: Validation seeking. Exploitation examples: Compliment-based elicitation. Layer: 8 (individual). Related concepts: Liking bias. Mitigations: Flattery skepticism.",
    "category": "TTP"
  },

  "CAT-2022-168": {
    "id": "CAT-2022-168",
    "name": "Oblique Reference",
    "description": "Oblique Reference mentions topic indirectly to elicit discussion without suspicion. This TTP exploits curiosity. Mechanism: Indirect priming. Exploitation examples: Side mentions to draw out info. Layer: 8 (individual). Related concepts: Elicitation. Mitigations: Topic avoidance.",
    "category": "TTP"
  },

  "CAT-2022-167": {
    "id": "CAT-2022-167",
    "name": "Neuro-Linguistic Programming",
    "description": "Neuro-Linguistic Programming claims to exploit neurological-linguistic connections for behavior influence. This exploit uses language patterns for rapport and suggestion. Mechanism: Modeled communication techniques. Exploitation examples: Persuasion training. Layer: 8 (individual). Related concepts: Rapport building. Mitigations: Critical language analysis.",
    "category": "Exploit"
  },

  "CAT-2022-166": {
    "id": "CAT-2022-166",
    "name": "Naive Mentality",
    "description": "Naive Mentality presents attacker as ignorant to elicit explanatory disclosure. This TTP exploits teaching instinct. Mechanism: Role reversal. Exploitation examples: \"Help me understand\" pretext. Layer: 8 (individual). Related concepts: Authority reversal. Mitigations: Information gating.",
    "category": "TTP"
  },

  "CAT-2022-165": {
    "id": "CAT-2022-165",
    "name": "Incentive Approach",
    "description": "Incentive Approach offers rewards for cooperation after discomfort. This TTP exploits relief seeking. Mechanism: Contrast between negative/positive states. Exploitation examples: Good cop/bad cop. Layer: 8 (individual). Related concepts: Relief manipulation. Mitigations: Incentive skepticism.",
    "category": "TTP"
  },

  "CAT-2022-164": {
    "id": "CAT-2022-164",
    "name": "Hour Glass Method",
    "description": "Hour Glass Method structures conversation benign → sensitive → benign to exploit memory primacy/recency. This TTP hides sensitive discussion. Mechanism: Serial position effect. Exploitation examples: Elicitation masking. Layer: 8 (individual). Related concepts: Memory bias. Mitigations: Full recall.",
    "category": "TTP"
  },

  "CAT-2022-163": {
    "id": "CAT-2022-163",
    "name": "Futility",
    "description": "Futility convinces subject resistance is pointless. This TTP exploits hopelessness. Mechanism: Perceived inevitability. Exploitation examples: \"It's over\" statements. Layer: 8 (individual). Related concepts: Learned helplessness. Mitigations: Hope maintenance.",
    "category": "TTP"
  },

  "CAT-2022-162": {
    "id": "CAT-2022-162",
    "name": "Flattery",
    "description": "Flattery compliments to win favor and reduce suspicion. This TTP exploits liking bias. Mechanism: Positive reinforcement. Exploitation examples: Social engineering rapport. Layer: 8 (individual). Related concepts: Halo effect. Mitigations: Compliment skepticism.",
    "category": "TTP"
  },

  "CAT-2022-161": {
    "id": "CAT-2022-161",
    "name": "File and Dossier",
    "description": "File and Dossier uses props (thick folders) to imply extensive knowledge. This TTP exploits perceived omniscience. Mechanism: Illusion of preparation. Exploitation examples: Interrogation intimidation. Layer: 8 (individual). Related concepts: Authority deception. Mitigations: Evidence demand.",
    "category": "TTP"
  },

  "CAT-2022-160": {
    "id": "CAT-2022-160",
    "name": "Fear-Up Approach",
    "description": "Fear-Up Approach increases subject fear to extract information. This TTP exploits threat perception. Mechanism: Stress-induced compliance. Exploitation examples: Threat-based interrogation. Layer: 8 (individual). Related concepts: Fear exploitation. Mitigations: Fear management.",
    "category": "TTP"
  },

  "CAT-2022-159": {
    "id": "CAT-2022-159",
    "name": "Fear-Down Approach",
    "description": "Fear-Down Approach calms subject to create safe disclosure environment. This TTP exploits relief. Mechanism: Stress reduction for openness. Exploitation examples: Rapport-based interrogation. Layer: 8 (individual). Related concepts: Trust building. Mitigations: Disclosure caution.",
    "category": "TTP"
  },

  "CAT-2022-158": {
    "id": "CAT-2022-158",
    "name": "Ethical Dilemma",
    "description": "Ethical Dilemma presents topic as moral conflict to elicit discussion. This TTP exploits conscience. Mechanism: Moral framing. Exploitation examples: Provoking debate on sensitive issues. Layer: 8 (individual). Related concepts: Moral baiting. Mitigations: Ethical clarity.",
    "category": "TTP"
  },

  "CAT-2022-157": {
    "id": "CAT-2022-157",
    "name": "Establish Your Identity",
    "description": "Establish Your Identity convinces subject they are mistaken for a criminal to provoke identity correction. This TTP exploits identity protection. Mechanism: False accusation pressure. Exploitation examples: Wrong person pretext. Layer: 8 (individual). Related concepts: Identity threat. Mitigations: Documentation.",
    "category": "TTP"
  },

  "CAT-2022-156": {
    "id": "CAT-2022-156",
    "name": "Emotional Approach",
    "description": "Emotional Approach identifies and exploits dominant emotion for disclosure. This TTP uses emotional leverage. Mechanism: Emotional state manipulation. Exploitation examples: Anger or sympathy induction. Layer: 8 (individual). Related concepts: Emotional hijacking. Mitigations: Emotional regulation.",
    "category": "TTP"
  },

  "CAT-2022-155": {
    "id": "CAT-2022-155",
    "name": "Elicitation of Information",
    "description": "Elicitation of Information uses prompts to gather target information. This TTP is core social engineering. Mechanism: Conversational cues. Exploitation examples: Casual questions revealing data. Layer: 8 (individual). Related concepts: Rapport, open questions. Mitigations: Information control.",
    "category": "TTP"
  },

  "CAT-2022-154": {
    "id": "CAT-2022-154",
    "name": "Disbelief",
    "description": "Disbelief feigns doubt to provoke clarification and disclosure. This TTP exploits need to convince. Mechanism: Challenging statements. Exploitation examples: \"I don't believe you\" prompting proof. Layer: 8 (individual). Related concepts: Provocation. Mitigations: Minimal response.",
    "category": "TTP"
  },

  "CAT-2022-153": {
    "id": "CAT-2022-153",
    "name": "Direct Approach",
    "description": "Direct Approach asks straight questions (most effective per studies). This TTP exploits simplicity. Mechanism: No deception needed. Exploitation examples: Straightforward interrogation. Layer: 8 (individual). Related concepts: Bluntness. Mitigations: Limited answers.",
    "category": "TTP"
  },

  "CAT-2022-152": {
    "id": "CAT-2022-152",
    "name": "Deliberate False Statement",
    "description": "Deliberate False Statement plants errors to exploit correction need. This TTP baits disclosure. Mechanism: Error provocation. Exploitation examples: Wrong facts to draw corrections. Layer: 8 (individual). Related concepts: Need to correct. Mitigations: Ignore bait.",
    "category": "TTP"
  },

  "CAT-2022-151": {
    "id": "CAT-2022-151",
    "name": "Criticism",
    "description": "Criticism attacks target to provoke defensive disclosure. This TTP exploits ego protection. Mechanism: Defensive reaction revealing info. Exploitation examples: Insults prompting justification. Layer: 8 (individual). Related concepts: Reactance. Mitigations: Non-response.",
    "category": "TTP"
  },

  "CAT-2022-150": {
    "id": "CAT-2022-150",
    "name": "Complaining-Tendency",
    "description": "Complaining-Tendency is habitual negative expression, exploitable for division. Mechanism: Venting reinforcement. Exploitation examples: Forums amplifying dissatisfaction. Layer: 8 (individual). Related concepts: Negativity bias. Mitigations: Positive focus.",
    "category": "Vulnerability"
  },

  "CAT-2022-149": {
    "id": "CAT-2022-149",
    "name": "Change of Scene",
    "description": "Change of Scene moves subject to new environment to alter mindset and increase openness. This TTP exploits context effects. Mechanism: Environmental priming. Exploitation examples: Interrogation room changes. Layer: 8 (individual). Related concepts: Context-dependent memory. Mitigations: Environmental awareness.",
    "category": "TTP"
  },

  "CAT-2022-148": {
    "id": "CAT-2022-148",
    "name": "Whaling",
    "description": "Whaling is spear-phishing against high-value targets (executives). This TTP exploits authority trust. Mechanism: Personalized high-stakes bait. Exploitation examples: CEO fraud. Layer: 8 (individual). Related concepts: Spear phishing. Mitigations: Executive training.",
    "category": "TTP"
  },

  "CAT-2022-147": {
    "id": "CAT-2022-147",
    "name": "Tailored Messaging",
    "description": "Tailored Messaging customizes content to target psychographics. This TTP exploits personalization. Mechanism: Relevance increasing engagement. Exploitation examples: Targeted ads/propaganda. Layer: 8 (individual). Related concepts: Microtargeting. Mitigations: Privacy tools.",
    "category": "TTP"
  },

  "CAT-2022-146": {
    "id": "CAT-2022-146",
    "name": "Sniper Ad Targeting",
    "description": "Sniper Ad Targeting delivers content to highly specific micro-audiences. This TTP exploits precision. Mechanism: Psychographic profiling. Exploitation examples: Political microtargeting. Layer: 8 (individual). Related concepts: Tailored messaging. Mitigations: Ad transparency.",
    "category": "TTP"
  },

  "CAT-2022-145": {
    "id": "CAT-2022-145",
    "name": "Pluridentity Attack",
    "description": "Pluridentity Attack assembles data from multiple sources to infer sensitive whole. This exploit targets fragmented privacy. Mechanism: Data fusion. Exploitation examples: Reconstructing phone numbers from partial leaks. Layer: 7-10. Related concepts: Inference vulnerability. Mitigations: Data minimization.",
    "category": "Exploit"
  },

  "CAT-2022-144": {
    "id": "CAT-2022-144",
    "name": "Inference Vulnerability",
    "description": "Inference Vulnerability allows sensitive information reconstruction from public fragments. This exploit targets data aggregation. Mechanism: Statistical inference. Exploitation examples: Doxxing from social media. Layer: 7-10. Related concepts: Data reconstruction. Mitigations: Information compartmentalization.",
    "category": "Exploit"
  },

  "CAT-2022-143": {
    "id": "CAT-2022-143",
    "name": "Data Vulnerability",
    "description": "Data Vulnerability is exposure of sensitive data. Mechanism: Poor hygiene. Exploitation examples: Data breaches. Layer: 8 (individual). Related concepts: Privacy paradox. Mitigations: Encryption.",
    "category": "Vulnerability"
  },

  "CAT-2022-142": {
    "id": "CAT-2022-142",
    "name": "Vishing",
    "description": "Vishing is voice-based phishing exploiting phone trust. This TTP bypasses email filters. Mechanism: Verbal authority. Exploitation examples: Bank impersonation calls. Layer: 8 (individual). Related concepts: Social engineering. Mitigations: Call verification.",
    "category": "TTP"
  },

  "CAT-2022-141": {
    "id": "CAT-2022-141",
    "name": "Spear Fishing",
    "description": "Spear Fishing is highly targeted phishing against specific individuals. This TTP exploits personalization. Mechanism: Research-based bait. Exploitation examples: Executive credential theft. Layer: 8 (individual). Related concepts: Whaling. Mitigations: Targeted training.",
    "category": "TTP"
  },

  "CAT-2022-140": {
    "id": "CAT-2022-140",
    "name": "Sock-Puppetry",
    "description": "Sock-Puppetry uses fake online personas for manipulation. This TTP exploits identity deception. Mechanism: Multiple false accounts. Exploitation examples: Astroturfing campaigns. Layer: 8 (individual). Related concepts: Impersonation. Mitigations: Account verification.",
    "category": "TTP"
  },

  "CAT-2022-139": {
    "id": "CAT-2022-139",
    "name": "SMSishing",
    "description": "SMSishing is phishing via SMS text messages. This TTP exploits mobile trust. Mechanism: Short message urgency. Exploitation examples: Fake delivery alerts. Layer: 8 (individual). Related concepts: Phishing variant. Mitigations: Link caution.",
    "category": "TTP"
  },

  "CAT-2022-138": {
    "id": "CAT-2022-138",
    "name": "Shilling Attack",
    "description": "Shilling Attack posts false reviews to manipulate ratings. This TTP exploits social proof. Mechanism: Artificial consensus. Exploitation examples: Product review manipulation. Layer: 8 (individual). Related concepts: Astroturfing. Mitigations: Review authenticity checks.",
    "category": "TTP"
  },

  "CAT-2022-137": {
    "id": "CAT-2022-137",
    "name": "Shill",
    "description": "Shill uses accomplices posing as independent to add credibility. This TTP exploits social proof. Mechanism: Third-party validation. Exploitation examples: Fake bidders in auctions. Layer: 8 (individual). Related concepts: Accomplice tactics. Mitigations: Independent verification.",
    "category": "TTP"
  },

  "CAT-2022-136": {
    "id": "CAT-2022-136",
    "name": "Semantic Attack",
    "description": "Semantic Attack mixes true and false information for confusion. This exploit targets truth discernment. Mechanism: Ambiguity and contradiction. Exploitation examples: Mixed fact/falsehood campaigns. Layer: 7-9. Related concepts: Grey propaganda. Mitigations: Source separation.",
    "category": "Exploit"
  },

  "CAT-2022-135": {
    "id": "CAT-2022-135",
    "name": "Scambaiting",
    "description": "Scambaiting wastes scammer time/resources (often with counter-malware). This defensive TTP exploits attacker persistence. Mechanism: Reverse social engineering. Exploitation examples: 419 scam reversals. Layer: 8 (individual). Related concepts: Vigilante defense. Mitigations: N/A (defensive).",
    "category": "TTP"
  },

  "CAT-2022-134": {
    "id": "CAT-2022-134",
    "name": "Reverse Social Engineering",
    "description": "Reverse Social Engineering creates situations where target seeks attacker's \"help\". This TTP exploits assistance bias. Mechanism: Manufactured need. Exploitation examples: Fake tech support sites. Layer: 8 (individual). Related concepts: Baiting reversal. Mitigations: Unsolicited help caution.",
    "category": "TTP"
  },

  "CAT-2022-133": {
    "id": "CAT-2022-133",
    "name": "Pretexting",
    "description": "Pretexting adopts cover story for access/information. This TTP is core social engineering. Mechanism: False narrative for legitimacy. Exploitation examples: Fake surveys or audits. Layer: 8 (individual). Related concepts: Impersonation. Mitigations: Identity verification.",
    "category": "TTP"
  },

  "CAT-2022-132": {
    "id": "CAT-2022-132",
    "name": "Phishing",
    "description": "Phishing fraudulently obtains sensitive information by impersonating trustworthy entities. This TTP exploits trust and urgency. Mechanism: Deceptive communication. Exploitation examples: Fake login pages. Layer: 8 (individual). Related concepts: Social engineering. Mitigations: Link verification.",
    "category": "TTP"
  },

  "CAT-2022-131": {
    "id": "CAT-2022-131",
    "name": "Impersonation Scam",
    "description": "Impersonation Scam poses as authority to extract money/information. This TTP exploits authority deference. Mechanism: False legitimacy. Exploitation examples: IRS or police scams. Layer: 8 (individual). Related concepts: Authority exploitation. Mitigations: Official channel use.",
    "category": "TTP"
  },

  "CAT-2022-130": {
    "id": "CAT-2022-130",
    "name": "Honey Trap",
    "description": "Honey Trap uses attractive lures to ensnare targets for exploitation. This TTP exploits desire/trust. Mechanism: Romantic or sexual bait. Exploitation examples: Espionage or blackmail. Layer: 8 (individual). Related concepts: Romance scam. Mitigations: Relationship caution.",
    "category": "TTP"
  },

  "CAT-2022-129": {
    "id": "CAT-2022-129",
    "name": "Honey Token",
    "description": "Honey Token is fictitious data to detect breaches. This defensive TTP exploits attacker greed. Mechanism: Alert on access. Exploitation examples: Fake credentials. Layer: 9 (group). Related concepts: Deception defense. Mitigations: N/A (defensive).",
    "category": "TTP"
  },

  "CAT-2022-128": {
    "id": "CAT-2022-128",
    "name": "Honey Pot",
    "description": "Honey Pot is attractive lure to draw attackers for monitoring. This defensive TTP exploits curiosity. Mechanism: Simulated vulnerable system. Exploitation examples: Decoy servers. Layer: 9 (group). Related concepts: Deception. Mitigations: N/A (defensive).",
    "category": "TTP"
  },

  "CAT-2022-127": {
    "id": "CAT-2022-127",
    "name": "Honey Phish",
    "description": "Honey Phish auto-replies to phishing with malicious links. This defensive TTP reverses attack. Mechanism: Counter-phishing. Exploitation examples: Reverse reconnaissance. Layer: 8 (individual). Related concepts: Scambaiting. Mitigations: N/A (defensive).",
    "category": "TTP"
  },

  "CAT-2022-126": {
    "id": "CAT-2022-126",
    "name": "Greenwashing",
    "description": "Greenwashing falsely claims environmental benefits to win favor. This TTP exploits virtue signaling. Mechanism: False sustainability claims. Exploitation examples: Fake eco-friendly marketing. Layer: 8 (individual). Related concepts: Virtue exploitation. Mitigations: Certification checks.",
    "category": "TTP"
  },

  "CAT-2022-125": {
    "id": "CAT-2022-125",
    "name": "Gaslighting",
    "description": "Gaslighting denies observable reality to sow doubt. This TTP exploits perception stability. Mechanism: Reality distortion. Exploitation examples: Abusive relationships or disinformation. Layer: 8 (individual). Related concepts: Psychological manipulation. Mitigations: External validation.",
    "category": "TTP"
  },

  "CAT-2022-124": {
    "id": "CAT-2022-124",
    "name": "Cognitive Malware Injection",
    "description": "Cognitive Malware Injection introduces contradictory information to degrade team performance. This TTP exploits consensus needs. Mechanism: Internal conflict creation. Exploitation examples: Disinformation in command chains. Layer: 9 (group). Related concepts: Divide and conquer. Mitigations: Information verification.",
    "category": "TTP"
  },

  "CAT-2022-123": {
    "id": "CAT-2022-123",
    "name": "Clone Phishing",
    "description": "Clone Phishing copies legitimate emails, modifying malicious elements. This TTP exploits familiarity. Mechanism: Near-identical replication. Exploitation examples: Modified legitimate thread hijacking. Layer: 8 (individual). Related concepts: Phishing variant. Mitigations: Header checks.",
    "category": "TTP"
  },

  "CAT-2022-122": {
    "id": "CAT-2022-122",
    "name": "Astroturfing",
    "description": "Astroturfing creates fake grassroots support. This TTP exploits social proof. Mechanism: Simulated consensus. Exploitation examples: Fake petition campaigns. Layer: 8 (individual). Related concepts: Manufactured opinion. Mitigations: Source tracing.",
    "category": "TTP"
  },

  "CAT-2022-121": {
    "id": "CAT-2022-121",
    "name": "Transmission Error",
    "description": "Transmission Error is errors in information passing. Mechanism: Communication breakdowns. Exploitation examples: Misinformation spread. Layer: 8 (individual). Related concepts: Telephone game. Mitigations: Verification.",
    "category": "Vulnerability"
  },

  "CAT-2022-120": {
    "id": "CAT-2022-120",
    "name": "Mis-Addressed Email",
    "description": "Mis-Addressed Email is sending to wrong recipient. Mechanism: Human error. Exploitation examples: Data leaks. Layer: 8 (individual). Related concepts: Typo squatting. Mitigations: Address confirmation.",
    "category": "Vulnerability"
  },

  "CAT-2022-119": {
    "id": "CAT-2022-119",
    "name": "Loss Error",
    "description": "Loss Error is data loss. Mechanism: Storage failures. Exploitation examples: Ransomware. Layer: 8 (individual). Related concepts: Backup neglect. Mitigations: Redundancy.",
    "category": "Vulnerability"
  },

  "CAT-2022-118": {
    "id": "CAT-2022-118",
    "name": "Leakage Errors",
    "description": "Leakage Errors are unintended data exposure. Mechanism: Oversight. Exploitation examples: Metadata leaks. Layer: 8 (individual). Related concepts: Side-channel attacks. Mitigations: Sanitization.",
    "category": "Vulnerability"
  },

  "CAT-2022-117": {
    "id": "CAT-2022-117",
    "name": "Disposal Errors",
    "description": "Disposal Errors are improper data destruction. Mechanism: Incomplete deletion. Exploitation examples: Dumpster diving. Layer: 8 (individual). Related concepts: Data remanence. Mitigations: Shredding.",
    "category": "Vulnerability"
  },

  "CAT-2022-116": {
    "id": "CAT-2022-116",
    "name": "Configuration Error",
    "description": "Configuration Error is misconfigured systems. Mechanism: Human setup mistakes. Exploitation examples: Open ports. Layer: 8 (individual). Related concepts: Default settings bias. Mitigations: Automation.",
    "category": "Vulnerability"
  },

  "CAT-2022-115": {
    "id": "CAT-2022-115",
    "name": "Curiosity",
    "description": "Curiosity is the drive to explore unknown. Mechanism: Information gap theory. Exploitation examples: Clickbait. Layer: 8 (individual). Related concepts: FOMO. Mitigations: Restraint.",
    "category": "Vulnerability"
  },

  "CAT-2022-114": {
    "id": "CAT-2022-114",
    "name": "Zeigarnik Effect",
    "description": "Zeigarnik Effect is better memory for incomplete tasks. Mechanism: Cognitive closure need. Exploitation examples: Cliffhangers in phishing. Layer: 8 (individual). Related concepts: Open loops. Mitigations: Task completion.",
    "category": "Vulnerability"
  },

  "CAT-2022-113": {
    "id": "CAT-2022-113",
    "name": "Whorfianism",
    "description": "Whorfianism is language shaping thought. Mechanism: Linguistic relativity. Exploitation examples: Euphemisms in propaganda. Layer: 8 (individual). Related concepts: Sapir-Whorf hypothesis. Mitigations: Multilingualism.",
    "category": "Vulnerability"
  },

  "CAT-2022-112": {
    "id": "CAT-2022-112",
    "name": "von Restorff Effect",
    "description": "von Restorff Effect is better recall of distinctive items. Mechanism: Isolation effect. Exploitation examples: Highlighting key disinformation. Layer: 8 (individual). Related concepts: Salience bias. Mitigations: Uniform presentation.",
    "category": "Vulnerability"
  },

  "CAT-2022-111": {
    "id": "CAT-2022-111",
    "name": "Unfinished Magnetizer",
    "description": "Unfinished Magnetizer presents incomplete information to occupy attention and memory. This exploit uses closure need. Mechanism: Zeigarnik effect. Exploitation examples: Open loops in messaging. Layer: 8 (individual). Related concepts: Incomplete task memory. Mitigations: Closure seeking.",
    "category": "Exploit"
  },

  "CAT-2022-110": {
    "id": "CAT-2022-110",
    "name": "Suggestion",
    "description": "Suggestion implants false memories or ideas. This TTP exploits suggestibility. Mechanism: Memory malleability. Exploitation examples: False witness planting. Layer: 8 (individual). Related concepts: False memory. Mitigations: Memory verification.",
    "category": "TTP"
  },

  "CAT-2022-109": {
    "id": "CAT-2022-109",
    "name": "Subjective Validation",
    "description": "Subjective Validation is accepting vague statements as personal. Mechanism: Barnum effect. Exploitation examples: Horoscopes. Layer: 8 (individual). Related concepts: Forer effect. Mitigations: Specificity testing.",
    "category": "Vulnerability"
  },

  "CAT-2022-108": {
    "id": "CAT-2022-108",
    "name": "Straw-Man Argument",
    "description": "Straw-Man Argument misrepresents opponent's position to easily refute. This TTP exploits misrepresentation. Mechanism: Distortion for victory illusion. Exploitation examples: Debate tactics. Layer: 8 (individual). Related concepts: Misrepresentation. Mitigations: Position clarification.",
    "category": "TTP"
  },

  "CAT-2022-107": {
    "id": "CAT-2022-107",
    "name": "Status Quo Bias",
    "description": "Status Quo Bias is preference for current state. Mechanism: Loss aversion. Exploitation examples: Resistance to change campaigns. Layer: 8 (individual). Related concepts: Inertia. Mitigations: Change framing.",
    "category": "Vulnerability"
  },

  "CAT-2022-106": {
    "id": "CAT-2022-106",
    "name": "Spotlight Effect",
    "description": "Spotlight Effect is overestimating others' attention. Mechanism: Egocentrism. Exploitation examples: Paranoia induction. Layer: 8 (individual). Related concepts: Imaginary audience. Mitigations: Perspective-taking.",
    "category": "Vulnerability"
  },

  "CAT-2022-105": {
    "id": "CAT-2022-105",
    "name": "Spacing Effect",
    "description": "Spacing Effect improves memory through repeated exposure over time. This exploit enhances message retention. Mechanism: Distributed practice. Exploitation examples: Repetition campaigns. Layer: 8 (individual). Related concepts: Repetition learning. Mitigations: Exposure control.",
    "category": "Exploit"
  },

  "CAT-2022-104": {
    "id": "CAT-2022-104",
    "name": "Source Monitoring Error",
    "description": "Source Monitoring Error is confusing memory sources. Mechanism: Source amnesia. Exploitation examples: Misattributed quotes in misinformation. Layer: 8 (individual). Related concepts: Cryptomnesia. Mitigations: Source tracking.",
    "category": "Vulnerability"
  },

  "CAT-2022-103": {
    "id": "CAT-2022-103",
    "name": "Serial Position Effect",
    "description": "Serial Position Effect is better recall of first/last items. Mechanism: Primacy/recency. Exploitation examples: Key messages at start/end. Layer: 8 (individual). Related concepts: Memory curves. Mitigations: Repetition.",
    "category": "Vulnerability"
  },

  "CAT-2022-102": {
    "id": "CAT-2022-102",
    "name": "Self-Serving Bias",
    "description": "Self-Serving Bias is attributing success internally, failure externally. Mechanism: Ego protection. Exploitation examples: Victim-blaming narratives. Layer: 8 (individual). Related concepts: Attribution theory. Mitigations: Reflection.",
    "category": "Vulnerability"
  },

  "CAT-2022-101": {
    "id": "CAT-2022-101",
    "name": "Self-Relevance Effect",
    "description": "Self-Relevance Effect is better memory for self-related info. Mechanism: Self-reference. Exploitation examples: Personalized scams. Layer: 8 (individual). Related concepts: Self-schema. Mitigations: Impersonalization.",
    "category": "Vulnerability"
  },

  "CAT-2022-100": {
    "id": "CAT-2022-100",
    "name": "SEO Manipulation Effect",
    "description": "SEO Manipulation Effect influences beliefs via search result ordering. This exploit uses ranking bias. Mechanism: First-page dominance. Exploitation examples: Search poisoning. Layer: 8 (individual). Related concepts: Availability heuristic. Mitigations: Multiple search engines.",
    "category": "Exploit"
  },

  "CAT-2022-099": {
    "id": "CAT-2022-099",
    "name": "Satisficing",
    "description": "Satisficing is accepting 'good enough' solutions. Mechanism: Bounded rationality. Exploitation examples: Low-effort scams. Layer: 8 (individual). Related concepts: Heuristics. Mitigations: Thorough search.",
    "category": "Vulnerability"
  },

  "CAT-2022-098": {
    "id": "CAT-2022-098",
    "name": "Risk Homeostasis",
    "description": "Risk Homeostasis is adjusting behavior to maintain risk level. Mechanism: Wilde's theory. Exploitation examples: Safety illusions leading to risk-taking. Layer: 8 (individual). Related concepts: Moral hazard. Mitigations: Risk awareness.",
    "category": "Vulnerability"
  },

  "CAT-2022-097": {
    "id": "CAT-2022-097",
    "name": "Relativism",
    "description": "Relativism is judging by relative standards. Mechanism: Comparative thinking. Exploitation examples: Normalizing bad behavior. Layer: 8 (individual). Related concepts: Moral relativism. Mitigations: Absolute standards.",
    "category": "Vulnerability"
  },

  "CAT-2022-096": {
    "id": "CAT-2022-096",
    "name": "Probability Blindness",
    "description": "Probability Blindness is poor probability judgment. Mechanism: Innumeracy. Exploitation examples: Lottery scams. Layer: 8 (individual). Related concepts: Gambler's fallacy. Mitigations: Stats training.",
    "category": "Vulnerability"
  },

  "CAT-2022-095": {
    "id": "CAT-2022-095",
    "name": "Pre-Suasion",
    "description": "Pre-Suasion prepares target for persuasion using cues. This exploit primes receptivity. Mechanism: Attention direction before message. Exploitation examples: Environmental setup before sales. Layer: 8 (individual). Related concepts: Priming. Mitigations: Cue awareness.",
    "category": "Exploit"
  },

  "CAT-2022-094": {
    "id": "CAT-2022-094",
    "name": "Planning Fallacy",
    "description": "Planning Fallacy is underestimating task time. Mechanism: Optimism bias. Exploitation examples: Deadline traps. Layer: 8 (individual). Related concepts: Hofstadter's law. Mitigations: Historical data.",
    "category": "Vulnerability"
  },

  "CAT-2022-093": {
    "id": "CAT-2022-093",
    "name": "Peak-End Rule",
    "description": "Peak-End Rule is judging experiences by peak/end. Mechanism: Memory bias. Exploitation examples: Ending messages strongly. Layer: 8 (individual). Related concepts: Recency effect. Mitigations: Holistic recall.",
    "category": "Vulnerability"
  },

  "CAT-2022-092": {
    "id": "CAT-2022-092",
    "name": "Overconfidence",
    "description": "Overconfidence is excessive belief in own judgment. Mechanism: Metacognitive illusion. Exploitation examples: Confidence tricks. Layer: 8 (individual). Related concepts: Dunning-Kruger. Mitigations: Calibration.",
    "category": "Vulnerability"
  },

  "CAT-2022-091": {
    "id": "CAT-2022-091",
    "name": "Optimism Bias",
    "description": "Optimism Bias is expecting positive outcomes. Mechanism: Motivational bias. Exploitation examples: Risky investments. Layer: 8 (individual). Related concepts: Unrealistic optimism. Mitigations: Pessimism planning.",
    "category": "Vulnerability"
  },

  "CAT-2022-090": {
    "id": "CAT-2022-090",
    "name": "Omission Bias",
    "description": "Omission Bias is preferring harms of inaction. Mechanism: Status quo preference. Exploitation examples: Vaccine hesitancy. Layer: 8 (individual). Related concepts: Default bias. Mitigations: Action framing.",
    "category": "Vulnerability"
  },

  "CAT-2022-089": {
    "id": "CAT-2022-089",
    "name": "Next-in-Line Effect",
    "description": "Next-in-Line Effect is diminished recall for previous speaker when next to speak. Mechanism: Self-focus during preparation. Exploitation examples: Public speaking manipulation. Layer: 8 (individual). Related concepts: Self-relevance. Mitigations: Active listening.",
    "category": "Vulnerability"
  },

  "CAT-2022-088": {
    "id": "CAT-2022-088",
    "name": "Neglect of Probability",
    "description": "Neglect of Probability is ignoring base rates. Mechanism: Insensitivity to likelihood. Exploitation examples: Terror hype. Layer: 8 (individual). Related concepts: Base rate fallacy. Mitigations: Bayesian thinking.",
    "category": "Vulnerability"
  },

  "CAT-2022-087": {
    "id": "CAT-2022-087",
    "name": "Negativity Bias",
    "description": "Negativity Bias is stronger weight on negative info. Mechanism: Evolutionary survival. Exploitation examples: Fear-mongering. Layer: 8 (individual). Related concepts: Loss aversion. Mitigations: Positive focus.",
    "category": "Vulnerability"
  },

  "CAT-2022-086": {
    "id": "CAT-2022-086",
    "name": "Narrative Influence",
    "description": "Narrative Influence uses stories to shape beliefs and behavior. This exploit bypasses rational analysis. Mechanism: Story immersion reducing critical thinking. Exploitation examples: Propaganda storytelling. Layer: 8 (individual). Related concepts: Transportation theory. Mitigations: Narrative deconstruction.",
    "category": "Exploit"
  },

  "CAT-2022-085": {
    "id": "CAT-2022-085",
    "name": "Mystery Magnetizer",
    "description": "Mystery Magnetizer provokes curiosity with incomplete information. This exploit occupies attention. Mechanism: Information gap driving engagement. Exploitation examples: Teaser campaigns. Layer: 8 (individual). Related concepts: Curiosity drive. Mitigations: Mystery resistance.",
    "category": "Exploit"
  },

  "CAT-2022-084": {
    "id": "CAT-2022-084",
    "name": "Mother Teresa Effect",
    "description": "Mother Teresa Effect is mood improvement from helping. Mechanism: Altruism reward. Exploitation examples: Charity scams. Layer: 8 (individual). Related concepts: Helper's high. Mitigations: Verify causes.",
    "category": "Vulnerability"
  },

  "CAT-2022-083": {
    "id": "CAT-2022-083",
    "name": "Mood-Congruent Memory",
    "description": "Mood-Congruent Memory is better recall in matching mood. Mechanism: State-dependent learning. Exploitation examples: Negative moods amplifying bad memories. Layer: 8 (individual). Related concepts: Mood bias. Mitigations: Mood management.",
    "category": "Vulnerability"
  },

  "CAT-2022-082": {
    "id": "CAT-2022-082",
    "name": "Mental Set",
    "description": "Mental Set is fixed approach to problems. Mechanism: Perseveration. Exploitation examples: Locked-in scams. Layer: 8 (individual). Related concepts: Functional fixedness. Mitigations: Lateral thinking.",
    "category": "Vulnerability"
  },

  "CAT-2022-081": {
    "id": "CAT-2022-081",
    "name": "Malware-Induced Misperception Attack",
    "description": "Malware-Induced Misperception Attack alters displayed messages to manipulate perception. This TTP exploits trust in interfaces. Mechanism: Text replacement for confusion. Exploitation examples: Fake alerts. Layer: 8 (individual). Related concepts: UI manipulation. Mitigations: System integrity checks.",
    "category": "TTP"
  },

  "CAT-2022-080": {
    "id": "CAT-2022-080",
    "name": "Self-Relevance Magnetizer",
    "description": "Self-Relevance Magnetizer presents self-related info to focus attention. This exploit uses personal relevance. Mechanism: Self-reference effect. Exploitation examples: Personalized phishing. Layer: 8 (individual). Related concepts: Self-schema. Mitigations: Impersonal processing.",
    "category": "Exploit"
  },

  "CAT-2022-079": {
    "id": "CAT-2022-079",
    "name": "Loss Aversion",
    "description": "Loss Aversion is stronger pain from losses. Mechanism: Prospect theory. Exploitation examples: Sunk cost fallacies. Layer: 8 (individual). Related concepts: Endowment effect. Mitigations: Framing gains.",
    "category": "Vulnerability"
  },

  "CAT-2022-078": {
    "id": "CAT-2022-078",
    "name": "Levels-of-Processing Effect",
    "description": "Levels-of-Processing Effect is deeper processing better memory. Mechanism: Craik-Lockhart. Exploitation examples: Semantic tricks. Layer: 8 (individual). Related concepts: Elaboration likelihood. Mitigations: Surface-level caution.",
    "category": "Vulnerability"
  },

  "CAT-2022-077": {
    "id": "CAT-2022-077",
    "name": "Leveling and Sharpening",
    "description": "Leveling and Sharpening is memory distortion over time. Mechanism: Serial reproduction. Exploitation examples: Rumor evolution. Layer: 8 (individual). Related concepts: Telephone game. Mitigations: Written records.",
    "category": "Vulnerability"
  },

  "CAT-2022-076": {
    "id": "CAT-2022-076",
    "name": "Involuntary Musical Imagery",
    "description": "Involuntary Musical Imagery is earworms occupying cognition. This exploit uses auditory loops. Mechanism: Auditory memory replay. Exploitation examples: Jingle marketing. Layer: 8 (individual). Related concepts: Earworm. Mitigations: Counter-melodies.",
    "category": "Exploit"
  },

  "CAT-2022-075": {
    "id": "CAT-2022-075",
    "name": "Involuntary Memory",
    "description": "Involuntary Memory is unintentional recollection triggered by cues. This vulnerability exposes past experiences. Mechanism: Associative recall. Exploitation examples: Trigger-based manipulation. Layer: 8 (individual). Related concepts: Proust effect. Mitigations: Trigger awareness.",
    "category": "Vulnerability"
  },

  "CAT-2022-074": {
    "id": "CAT-2022-074",
    "name": "Inoculation Effect",
    "description": "Inoculation Effect strengthens resistance through pre-exposure to weakened threats. This defensive exploit builds immunity. Mechanism: Counter-argument preparation. Exploitation examples: Prebunking disinformation. Layer: 8 (individual). Related concepts: Psychological vaccination. Mitigations: N/A (defensive).",
    "category": "Exploit"
  },

  "CAT-2022-073": {
    "id": "CAT-2022-073",
    "name": "Illusory Correlation",
    "description": "Illusory Correlation is perceiving nonexistent relationships. Mechanism: Apophenia. Exploitation examples: Conspiracy theories. Layer: 8 (individual). Related concepts: Pareidolia. Mitigations: Statistical checks.",
    "category": "Vulnerability"
  },

  "CAT-2022-072": {
    "id": "CAT-2022-072",
    "name": "Illusion of Control",
    "description": "Illusion of Control is overestimating personal influence. Mechanism: Agency bias. Exploitation examples: Gambling illusions. Layer: 8 (individual). Related concepts: Locus of control. Mitigations: Randomness awareness.",
    "category": "Vulnerability"
  },

  "CAT-2022-071": {
    "id": "CAT-2022-071",
    "name": "IKEA Effect",
    "description": "IKEA Effect is valuing self-made items more. Mechanism: Effort justification. Exploitation examples: DIY scams. Layer: 8 (individual). Related concepts: Sunk cost. Mitigations: Objective valuation.",
    "category": "Vulnerability"
  },

  "CAT-2022-070": {
    "id": "CAT-2022-070",
    "name": "Hyperbolic Discounting",
    "description": "Hyperbolic Discounting is preferring immediate rewards. Mechanism: Temporal discounting. Exploitation examples: Impulse buys. Layer: 8 (individual). Related concepts: Present bias. Mitigations: Commitment devices.",
    "category": "Vulnerability"
  },

  "CAT-2022-069": {
    "id": "CAT-2022-069",
    "name": "Hindsight Bias",
    "description": "Hindsight Bias is seeing past events as predictable. Mechanism: Knew-it-all-along. Exploitation examples: Post-event manipulation. Layer: 8 (individual). Related concepts: Retrospective bias. Mitigations: Pre-registration.",
    "category": "Vulnerability"
  },

  "CAT-2022-068": {
    "id": "CAT-2022-068",
    "name": "Halo Effect",
    "description": "Halo Effect is overall impression influencing specifics. Mechanism: Cognitive generalization. Exploitation examples: Celebrity endorsements. Layer: 8 (individual). Related concepts: Horn effect. Mitigations: Trait separation.",
    "category": "Vulnerability"
  },

  "CAT-2022-067": {
    "id": "CAT-2022-067",
    "name": "Gambler's Fallacy",
    "description": "Gambler's Fallacy is expecting reversal after streak. Mechanism: Independence of events. Exploitation examples: Lottery myths. Layer: 8 (individual). Related concepts: Monte Carlo fallacy. Mitigations: Probability education.",
    "category": "Vulnerability"
  },

  "CAT-2022-066": {
    "id": "CAT-2022-066",
    "name": "Fundamental Attribution Error",
    "description": "Fundamental Attribution Error is overemphasizing disposition. Mechanism: Attribution theory. Exploitation examples: Character assassinations. Layer: 8 (individual). Related concepts: Actor-observer bias. Mitigations: Situational awareness.",
    "category": "Vulnerability"
  },

  "CAT-2022-065": {
    "id": "CAT-2022-065",
    "name": "Functional Fixedness",
    "description": "Functional Fixedness is seeing objects only in usual function. Mechanism: Mental set. Exploitation examples: Improvised weapon oversight. Layer: 8 (individual). Related concepts: Einstellung effect. Mitigations: Creative thinking.",
    "category": "Vulnerability"
  },

  "CAT-2022-064": {
    "id": "CAT-2022-064",
    "name": "Frequency Illusion",
    "description": "Frequency Illusion is noticing something more after awareness. Mechanism: Baader-Meinhof. Exploitation examples: Targeted ads. Layer: 8 (individual). Related concepts: Selection bias. Mitigations: Attention tracking.",
    "category": "Vulnerability"
  },

  "CAT-2022-063": {
    "id": "CAT-2022-063",
    "name": "Framing Effect",
    "description": "Framing Effect is decision influenced by presentation. Mechanism: Prospect theory. Exploitation examples: Positive/negative spin. Layer: 8 (individual). Related concepts: Anchoring. Mitigations: Reframing exercises.",
    "category": "Vulnerability"
  },

  "CAT-2022-062": {
    "id": "CAT-2022-062",
    "name": "Foot-in-the-Door Technique",
    "description": "Foot-in-the-Door Technique gains compliance with small request then larger. This TTP exploits consistency. Mechanism: Commitment escalation. Exploitation examples: Sales or donation upsell. Layer: 8 (individual). Related concepts: Commitment consistency. Mitigations: Request scrutiny.",
    "category": "TTP"
  },

  "CAT-2022-061": {
    "id": "CAT-2022-061",
    "name": "Fear of Missing Out",
    "description": "Fear of Missing Out drives action from perceived exclusion. This vulnerability combines scarcity and social proof. Mechanism: Social comparison anxiety. Exploitation examples: Limited offers. Layer: 8 (individual). Related concepts: Scarcity, social proof. Mitigations: Abundance mindset.",
    "category": "Exploit"
  },

  "CAT-2022-060": {
    "id": "CAT-2022-060",
    "name": "False Uniqueness Bias",
    "description": "False Uniqueness Bias is overestimating own rarity. Mechanism: Ego-centrism. Exploitation examples: Exclusive scam offers. Layer: 8 (individual). Related concepts: False consensus. Mitigations: Comparative data.",
    "category": "Vulnerability"
  },

  "CAT-2022-059": {
    "id": "CAT-2022-059",
    "name": "False Memory",
    "description": "False Memory is recalling nonexistent events. Mechanism: Suggestibility. Exploitation examples: Eyewitness manipulation. Layer: 8 (individual). Related concepts: Mandela effect. Mitigations: Source monitoring.",
    "category": "Vulnerability"
  },

  "CAT-2022-058": {
    "id": "CAT-2022-058",
    "name": "False Consensus Effect",
    "description": "False Consensus Effect is overestimating agreement. Mechanism: Projection bias. Exploitation examples: Echo chamber amplification. Layer: 8 (individual). Related concepts: Pluralistic ignorance. Mitigations: Diverse polling.",
    "category": "Vulnerability"
  },

  "CAT-2022-057": {
    "id": "CAT-2022-057",
    "name": "Endowment Effect",
    "description": "Endowment Effect is valuing owned items more. Mechanism: Ownership bias. Exploitation examples: Auction traps. Layer: 8 (individual). Related concepts: Loss aversion. Mitigations: Willingness-to-pay tests.",
    "category": "Vulnerability"
  },

  "CAT-2022-056": {
    "id": "CAT-2022-056",
    "name": "Egocentric Bias",
    "description": "Egocentric Bias is self-centered perspective. Mechanism: Anchoring on own view. Exploitation examples: Empathy gaps in scams. Layer: 8 (individual). Related concepts: Curse of knowledge. Mitigations: Perspective-taking.",
    "category": "Vulnerability"
  },

  "CAT-2022-055": {
    "id": "CAT-2022-055",
    "name": "Ear Worm",
    "description": "Ear Worm is catchy tunes occupying cognition. This exploit uses auditory memory. Mechanism: Involuntary replay. Exploitation examples: Viral jingles. Layer: 8 (individual). Related concepts: Involuntary musical imagery. Mitigations: Counter-tunes.",
    "category": "Exploit"
  },

  "CAT-2022-054": {
    "id": "CAT-2022-054",
    "name": "Dunning–Kruger Effect",
    "description": "Dunning–Kruger Effect is incompetents overestimating ability. Mechanism: Metacognitive failure. Exploitation examples: Dunning targets in cons. Layer: 8 (individual). Related concepts: Overconfidence. Mitigations: Expertise feedback.",
    "category": "Vulnerability"
  },

  "CAT-2022-053": {
    "id": "CAT-2022-053",
    "name": "Dread Aversion",
    "description": "Dread Aversion is avoiding feared outcomes disproportionately. Mechanism: Anticipatory anxiety. Exploitation examples: Insurance scams. Layer: 8 (individual). Related concepts: Risk aversion. Mitigations: Probability calibration.",
    "category": "Vulnerability"
  },

  "CAT-2022-052": {
    "id": "CAT-2022-052",
    "name": "Default Bias",
    "description": "Default Bias is preferring pre-selected options. Mechanism: Inertia. Exploitation examples: Opt-out traps. Layer: 8 (individual). Related concepts: Status quo bias. Mitigations: Active choice designs.",
    "category": "Vulnerability"
  },

  "CAT-2022-051": {
    "id": "CAT-2022-051",
    "name": "Decoy Effect",
    "description": "Decoy Effect switches preference with inferior third option. This exploit manipulates choice architecture. Mechanism: Asymmetrical dominance. Exploitation examples: Pricing tiers. Layer: 8 (individual). Related concepts: Choice overload. Mitigations: Option comparison.",
    "category": "Exploit"
  },

  "CAT-2022-050": {
    "id": "CAT-2022-050",
    "name": "Decision Fatigue",
    "description": "Decision Fatigue is declining decision quality after many choices. This vulnerability reduces resistance. Mechanism: Ego depletion. Exploitation examples: Late-day sales pressure. Layer: 8 (individual). Related concepts: Willpower depletion. Mitigations: Decision timing.",
    "category": "Vulnerability"
  },

  "CAT-2022-049": {
    "id": "CAT-2022-049",
    "name": "Context Dependent Memory",
    "description": "Context Dependent Memory is better recall in original context. Mechanism: Encoding specificity. Exploitation examples: Trigger-based recalls in manipulation. Layer: 8 (individual). Related concepts: State-dependent learning. Mitigations: Diverse encoding.",
    "category": "Vulnerability"
  },

  "CAT-2022-048": {
    "id": "CAT-2022-048",
    "name": "Confirmation Bias",
    "description": "Confirmation Bias is seeking confirming evidence. Mechanism: Motivated reasoning. Exploitation examples: Echo chambers. Layer: 8 (individual). Related concepts: Selective exposure. Mitigations: Disconfirming search.",
    "category": "Vulnerability"
  },

  "CAT-2022-047": {
    "id": "CAT-2022-047",
    "name": "Cognitive Malware",
    "description": "Cognitive Malware introduces conflicting information to degrade performance. This exploit targets consensus. Mechanism: Internal conflict. Exploitation examples: Team disinformation. Layer: 9 (group). Related concepts: Divide and conquer. Mitigations: Verification.",
    "category": "Exploit"
  },

  "CAT-2022-046": {
    "id": "CAT-2022-046",
    "name": "Cognitive Dissonance",
    "description": "Cognitive Dissonance is discomfort from conflicting beliefs. Mechanism: Festinger theory. Exploitation examples: Forced compliance. Layer: 8 (individual). Related concepts: Justification. Mitigations: Dissonance resolution.",
    "category": "Vulnerability"
  },

  "CAT-2022-045": {
    "id": "CAT-2022-045",
    "name": "Cognitive Deception",
    "description": "Cognitive Deception creates illusions aiding self-deception. This exploit mirrors perceptual illusions. Mechanism: Internal bias reinforcement. Exploitation examples: Rationalization of harmful beliefs. Layer: 8 (individual). Related concepts: Motivated reasoning. Mitigations: External validation.",
    "category": "Exploit"
  },

  "CAT-2022-044": {
    "id": "CAT-2022-044",
    "name": "Clustering Illusion",
    "description": "Clustering Illusion is seeing patterns in random data. Mechanism: Apophenia. Exploitation examples: Conspiracy theories. Layer: 8 (individual). Related concepts: Pareidolia. Mitigations: Statistical testing.",
    "category": "Vulnerability"
  },

  "CAT-2022-043": {
    "id": "CAT-2022-043",
    "name": "Classical Conditioning",
    "description": "Classical Conditioning associates neutral stimulus with potent one for desired response. This exploit builds automatic reactions. Mechanism: Pavlovian pairing. Exploitation examples: Brand-association advertising. Layer: 8 (individual). Related concepts: Associative learning. Mitigations: Awareness training.",
    "category": "Exploit"
  },

  "CAT-2022-042": {
    "id": "CAT-2022-042",
    "name": "Ben Franklin Effect",
    "description": "Ben Franklin Effect increases liking after doing favor. This exploit uses reciprocity reversal. Mechanism: Cognitive dissonance reduction through rationalization. Exploitation examples: Requesting help to build rapport. Layer: 8 (individual). Related concepts: Reciprocity variant. Mitigations: Favor awareness.",
    "category": "Exploit"
  },

  "CAT-2022-041": {
    "id": "CAT-2022-041",
    "name": "Belief Bias",
    "description": "Belief Bias is judging arguments by belief consistency. Mechanism: Prior bias. Exploitation examples: Partisan news. Layer: 8 (individual). Related concepts: Motivated reasoning. Mitigations: Logic training.",
    "category": "Vulnerability"
  },

  "CAT-2022-040": {
    "id": "CAT-2022-040",
    "name": "Base Rate Neglect",
    "description": "Base Rate Neglect is ignoring statistical base rates. Mechanism: Representativeness heuristic. Exploitation examples: Rare event hype. Layer: 8 (individual). Related concepts: Bayes' theorem. Mitigations: Base rate reminders.",
    "category": "Vulnerability"
  },

  "CAT-2022-039": {
    "id": "CAT-2022-039",
    "name": "Barnum Statement",
    "description": "Barnum Statement uses vague descriptions accepted as personal. This TTP exploits subjective validation. Mechanism: Generalities fitting many. Exploitation examples: Fortune telling. Layer: 8 (individual). Related concepts: Forer effect. Mitigations: Specificity demand.",
    "category": "TTP"
  },

  "CAT-2022-038": {
    "id": "CAT-2022-038",
    "name": "Availability Heuristic",
    "description": "Availability Heuristic judges likelihood by recall ease. This vulnerability is influenced by recency/salience. Mechanism: Memory accessibility bias. Exploitation examples: Recent event hype. Layer: 8 (individual). Related concepts: Recency effect. Mitigations: Statistical thinking.",
    "category": "Vulnerability"
  },

  "CAT-2022-037": {
    "id": "CAT-2022-037",
    "name": "Anchoring",
    "description": "Anchoring focuses on first information for decisions. This vulnerability allows initial value manipulation. Mechanism: Insufficient adjustment from anchor. Exploitation examples: High initial prices. Layer: 8 (individual). Related concepts: Priming. Mitigations: Multiple anchors.",
    "category": "Vulnerability"
  },

  "CAT-2022-036": {
    "id": "CAT-2022-036",
    "name": "Ambiguous Self-Induced Disinformation Attack",
    "description": "Ambiguous Self-Induced Disinformation Attack introduces ambiguous information causing self-reinforcing misinterpretation. This TTP exploits uncertainty. Mechanism: Target fills gaps with bias. Exploitation examples: Vague rumors. Layer: 8 (individual). Related concepts: Motivated reasoning. Mitigations: Clarification seeking.",
    "category": "TTP"
  },

  "CAT-2022-035": {
    "id": "CAT-2022-035",
    "name": "Ambient Tactical Deception Attack",
    "description": "Ambient Tactical Deception Attack replaces message text with ambiguous/opposite content. This TTP exploits trust in communication. Mechanism: Man-in-the-middle text alteration. Exploitation examples: Chat manipulation. Layer: 8 (individual). Related concepts: Interception. Mitigations: End-to-end encryption.",
    "category": "TTP"
  },

  "CAT-2022-034": {
    "id": "CAT-2022-034",
    "name": "Actor-Observer Bias",
    "description": "Actor-Observer Bias attributes own actions to situation, others to disposition. Mechanism: Attribution asymmetry. Exploitation examples: Blame shifting. Layer: 8 (individual). Related concepts: Fundamental attribution error. Mitigations: Perspective swap.",
    "category": "Vulnerability"
  },

  "CAT-2022-033": {
    "id": "CAT-2022-033",
    "name": "Wikjacking",
    "description": "Wikjacking inserts false information into Wikipedia for credibility. This TTP exploits authority perception. Mechanism: Crowdsourced trust. Exploitation examples: Fake citations. Layer: 8 (individual). Related concepts: Authority deception. Mitigations: Source verification.",
    "category": "TTP"
  },

  "CAT-2022-032": {
    "id": "CAT-2022-032",
    "name": "Wi-Fi Evil Twin",
    "description": "Wi-Fi Evil Twin impersonates legitimate access points for interception. This TTP exploits network trust. Mechanism: Spoofed SSID. Exploitation examples: Public WiFi attacks. Layer: 8 (individual). Related concepts: Man-in-the-middle. Mitigations: VPN usage.",
    "category": "TTP"
  },

  "CAT-2022-031": {
    "id": "CAT-2022-031",
    "name": "Water Hole Attack",
    "description": "Water Hole Attack infects frequently visited sites. This TTP exploits habit. Mechanism: Targeted compromise of trusted domains. Exploitation examples: Compromised news sites. Layer: 8 (individual). Related concepts: Supply chain attack. Mitigations: Site diversity.",
    "category": "TTP"
  },

  "CAT-2022-030": {
    "id": "CAT-2022-030",
    "name": "Typosquatting",
    "description": "Typosquatting registers misspelled domains for phishing. This TTP exploits attentional blindness. Mechanism: Visual similarity. Exploitation examples: amaz0n.com. Layer: 8 (individual). Related concepts: Homograph attack. Mitigations: Careful typing.",
    "category": "TTP"
  },

  "CAT-2022-029": {
    "id": "CAT-2022-029",
    "name": "Social Phishing",
    "description": "Social Phishing uses social media for phishing. This TTP exploits platform trust. Mechanism: Direct messaging attacks. Exploitation examples: Fake friend requests. Layer: 8 (individual). Related concepts: Phishing variant. Mitigations: Friend verification.",
    "category": "TTP"
  },

  "CAT-2022-028": {
    "id": "CAT-2022-028",
    "name": "Lateral Phishing",
    "description": "Lateral Phishing pivots from low to high-privilege targets internally. This TTP exploits internal trust. Mechanism: Compromised account usage. Exploitation examples: Forwarded phishing escalation. Layer: 9 (group). Related concepts: Insider threat. Mitigations: Internal verification.",
    "category": "TTP"
  },

  "CAT-2022-027": {
    "id": "CAT-2022-027",
    "name": "Fluency Effect",
    "description": "Fluency Effect is easier processing feels truer. Mechanism: Processing fluency. Exploitation examples: Repetition in lies. Layer: 8 (individual). Related concepts: Illusion of truth. Mitigations: Effortful thinking.",
    "category": "Vulnerability"
  },

  "CAT-2022-026": {
    "id": "CAT-2022-026",
    "name": "Familiarity",
    "description": "Familiarity is preference for known things. Mechanism: Mere exposure effect. Exploitation examples: Brand loyalty scams. Layer: 8 (individual). Related concepts: Recognition heuristic. Mitigations: Novelty exposure.",
    "category": "Vulnerability"
  },

  "CAT-2022-025": {
    "id": "CAT-2022-025",
    "name": "Brandjacking",
    "description": "Brandjacking acquires online identity to capitalize on reputation. This TTP exploits brand trust. Mechanism: Identity theft for credibility. Exploitation examples: Fake corporate accounts. Layer: 8 (individual). Related concepts: Impersonation. Mitigations: Brand monitoring.",
    "category": "TTP"
  },

  "CAT-2022-023": {
    "id": "CAT-2022-023",
    "name": "Trick Question",
    "description": "Trick Question uses confusing language to elicit unintended answers. This TTP exploits assumption. Mechanism: Ambiguous phrasing. Exploitation examples: Dark pattern opt-ins. Layer: 8 (individual). Related concepts: Leading question. Mitigations: Careful reading.",
    "category": "TTP"
  },

  "CAT-2022-022": {
    "id": "CAT-2022-022",
    "name": "Spam",
    "description": "Spam is mass unsolicited messaging for scale. This TTP exploits volume. Mechanism: Overwhelm filters. Exploitation examples: Email floods. Layer: 8 (individual). Related concepts: Firehose variant. Mitigations: Filtering.",
    "category": "TTP"
  },

  "CAT-2022-021": {
    "id": "CAT-2022-021",
    "name": "Sneak into Basket",
    "description": "Sneak into Basket adds items to purchase without notice. This TTP exploits attention gaps. Mechanism: Hidden additions. Exploitation examples: Pre-checked add-ons. Layer: 8 (individual). Related concepts: Dark patterns. Mitigations: Cart review.",
    "category": "TTP"
  },

  "CAT-2022-020": {
    "id": "CAT-2022-020",
    "name": "Scareware",
    "description": "Scareware presents false threats to prompt actions. This TTP exploits fear. Mechanism: Fake alerts. Exploitation examples: Rogue antivirus. Layer: 8 (individual). Related concepts: Fear exploitation. Mitigations: Official sources.",
    "category": "TTP"
  },

  "CAT-2022-019": {
    "id": "CAT-2022-019",
    "name": "Roach Motel",
    "description": "Roach Motel makes entry easy, exit hard. This TTP exploits inertia. Mechanism: Asymmetric design. Exploitation examples: Hard cancellations. Layer: 8 (individual). Related concepts: Dark patterns. Mitigations: Easy opt-out.",
    "category": "TTP"
  },

  "CAT-2022-018": {
    "id": "CAT-2022-018",
    "name": "Privacy Zuckering",
    "description": "Privacy Zuckering tricks into sharing more than intended. This TTP exploits interface confusion. Mechanism: Hidden settings. Exploitation examples: Default public posts. Layer: 8 (individual). Related concepts: Dark patterns. Mitigations: Privacy review.",
    "category": "TTP"
  },

  "CAT-2022-017": {
    "id": "CAT-2022-017",
    "name": "Price Comparison Prevention",
    "description": "Price Comparison Prevention obscures pricing for comparison. This TTP exploits cognitive effort. Mechanism: Unit variation. Exploitation examples: Different metrics. Layer: 8 (individual). Related concepts: Dark patterns. Mitigations: Standardized comparison.",
    "category": "TTP"
  },

  "CAT-2022-016": {
    "id": "CAT-2022-016",
    "name": "Persuasive Technology",
    "description": "Persuasive Technology designs systems to change attitudes/behavior. This TTP exploits behavioral nudging. Mechanism: Intentional design for influence. Exploitation examples: Addictive apps. Layer: 8 (individual). Related concepts: Nudging. Mitigations: Design awareness.",
    "category": "TTP"
  },

  "CAT-2022-015": {
    "id": "CAT-2022-015",
    "name": "Mouse-Trapping",
    "description": "Mouse-Trapping prevents site exit with pop-ups. This TTP exploits escape difficulty. Mechanism: Navigation blocking. Exploitation examples: Endless pop-ups. Layer: 8 (individual). Related concepts: Dark patterns. Mitigations: Browser controls.",
    "category": "TTP"
  },

  "CAT-2022-014": {
    "id": "CAT-2022-014",
    "name": "Misdirection Distraction",
    "description": "Misdirection Distraction directs attention away from key elements. This exploit uses salience control. Mechanism: Attention diversion. Exploitation examples: Magician tricks or scam diversions. Layer: 8 (individual). Related concepts: Sleight of hand. Mitigations: Focused attention.",
    "category": "Exploit"
  },

  "CAT-2022-013": {
    "id": "CAT-2022-013",
    "name": "Malvertising",
    "description": "Malvertising places malware in legitimate ads. This TTP exploits ad network trust. Mechanism: Compromised ad delivery. Exploitation examples: Drive-by downloads. Layer: 8 (individual). Related concepts: Supply chain attack. Mitigations: Ad blockers.",
    "category": "TTP"
  },

  "CAT-2022-012": {
    "id": "CAT-2022-012",
    "name": "Hidden Costs",
    "description": "Hidden Costs reveals fees late in process. This TTP exploits sunk cost. Mechanism: Gradual disclosure. Exploitation examples: Airline fees. Layer: 8 (individual). Related concepts: Dark patterns. Mitigations: Total cost upfront.",
    "category": "TTP"
  },

  "CAT-2022-011": {
    "id": "CAT-2022-011",
    "name": "Friend Spam",
    "description": "Friend Spam requests contact access to send invitations. This TTP exploits social connections. Mechanism: Apparent personal outreach. Exploitation examples: App permission abuse. Layer: 8 (individual). Related concepts: Social proof. Mitigations: Permission caution.",
    "category": "TTP"
  },

  "CAT-2022-010": {
    "id": "CAT-2022-010",
    "name": "Forced Continuity",
    "description": "Forced Continuity charges after free trial without notice. This TTP exploits inertia. Mechanism: Auto-renewal. Exploitation examples: Subscription traps. Layer: 8 (individual). Related concepts: Default bias. Mitigations: Explicit consent.",
    "category": "TTP"
  },

  "CAT-2022-009": {
    "id": "CAT-2022-009",
    "name": "Disguised Ads",
    "description": "Disguised Ads blend with content to trick clicks. This TTP exploits attention. Mechanism: Native advertising deception. Exploitation examples: Sponsored content. Layer: 8 (individual). Related concepts: Deceptive design. Mitigations: Ad labeling awareness.",
    "category": "TTP"
  },

  "CAT-2022-008": {
    "id": "CAT-2022-008",
    "name": "Dark Design Patterns",
    "description": "Dark Design Patterns use UI to guide undesired actions. This TTP exploits cognitive shortcuts. Mechanism: Intentional friction/asymmetry. Exploitation examples: Hard cancellations. Layer: 8 (individual). Related concepts: Nudging misuse. Mitigations: Ethical design.",
    "category": "TTP"
  },

  "CAT-2022-007": {
    "id": "CAT-2022-007",
    "name": "Confirm Shaming",
    "description": "Confirm Shaming guilts users into opting in. This TTP exploits social desirability. Mechanism: Negative labeling of decline. Exploitation examples: \"No, I hate saving\" buttons. Layer: 8 (individual). Related concepts: Guilt manipulation. Mitigations: Neutral options.",
    "category": "TTP"
  },

  "CAT-2022-006": {
    "id": "CAT-2022-006",
    "name": "Click-Baiting",
    "description": "Click-Baiting uses sensational headlines to drive clicks. This TTP exploits curiosity. Mechanism: Emotional trigger. Exploitation examples: Misleading titles. Layer: 8 (individual). Related concepts: Curiosity gap. Mitigations: Headline skepticism.",
    "category": "TTP"
  },

  "CAT-2022-005": {
    "id": "CAT-2022-005",
    "name": "Bait and Switch",
    "description": "Bait and Switch advertises one thing but delivers another. This TTP exploits expectation. Mechanism: Substitution after commitment. Exploitation examples: Hotel booking scams. Layer: 8 (individual). Related concepts: False advertising. Mitigations: Verification.",
    "category": "TTP"
  },

  "CAT-2022-004": {
    "id": "CAT-2022-004",
    "name": "Addictive Technology",
    "description": "Addictive Technology incorporates mechanisms to maximize engagement. This TTP exploits reward systems. Mechanism: Variable ratio reinforcement. Exploitation examples: Infinite scroll. Layer: 8 (individual). Related concepts: Behavioral addiction. Mitigations: Usage limits.",
    "category": "TTP"
  },

  "CAT-2022-003": {
    "id": "CAT-2022-003",
    "name": "Nudging",
    "description": "Nudging influences choices through design without restricting freedom. This TTP exploits default bias. Mechanism: Choice architecture. Exploitation examples: Organ donation opt-out. Layer: 8 (individual). Related concepts: Libertarian paternalism. Mitigations: Awareness of influence.",
    "category": "TTP"
  },

  "CAT-2022-002": {
    "id": "CAT-2022-002",
    "name": "Forcing Function",
    "description": "Forcing Function designs systems to prevent errors through constraint. This defensive TTP exploits human fallibility. Mechanism: Physical or procedural barriers. Exploitation examples: Seatbelt interlocks. Layer: 8 (individual). Related concepts: Error-proofing. Mitigations: N/A (defensive).",
    "category": "TTP"
  },

  "CAT-2022-001": {
    "id": "CAT-2022-001",
    "name": "Fogg Model of Behavior",
    "description": "Fogg Model of Behavior predicts action when motivation, ability, and prompt coincide. This framework exploits behavioral triggers. Mechanism: B=MAP formula. Exploitation examples: App notifications. Layer: 8 (individual). Related concepts: Behavior design. Mitigations: Trigger management.",
    "category": "Tool/TTP"
  },

  "CAT-2021-012": {
    "id": "CAT-2021-012",
    "name": "Reciprocation",
    "description": "Reciprocation exploits the social norm to return favors. This vulnerability creates obligation. Mechanism: Reciprocity principle. Exploitation examples: Unsolicited gifts leading to compliance. Layer: 8 (individual). Related concepts: Gift bias. Mitigations: No-obligation awareness.",
    "category": "Exploit"
  },

  "CAT-2021-011": {
    "id": "CAT-2021-011",
    "name": "Authority",
    "description": "Authority exploits deference to perceived experts or leaders. This vulnerability bypasses critical thinking. Mechanism: Obedience conditioning. Exploitation examples: Fake official requests. Layer: 8 (individual). Related concepts: Milgram effect. Mitigations: Authority verification.",
    "category": "Exploit"
  },

  "CAT-2021-010": {
    "id": "CAT-2021-010",
    "name": "Appeal to Excitement",
    "description": "Appeal to Excitement uses thrilling promises to lower defenses. This exploit targets impulsivity and boredom. Mechanism: Emotional arousal. Exploitation examples: Get-rich-quick schemes. Layer: 8 (individual). Related concepts: Novelty seeking. Mitigations: Excitement skepticism.",
    "category": "Exploit"
  },

  "CAT-2021-009": {
    "id": "CAT-2021-009",
    "name": "Low Agreeableness",
    "description": "Low Agreeableness makes individuals antagonistic and less cooperative. This vulnerability can be exploited with reverse psychology. Mechanism: Spite-driven choice. Exploitation examples: Provoking contrary action. Layer: 8 (individual). Related concepts: Reactance. Mitigations: Emotional regulation.",
    "category": "Vulnerability"
  },

  "CAT-2021-008": {
    "id": "CAT-2021-008",
    "name": "Low Extraversion",
    "description": "Low Extraversion reduces social seeking. This vulnerability can be exploited by gradual social introduction. Mechanism: Comfort with isolation. Exploitation examples: Slow grooming. Layer: 8 (individual). Related concepts: Introversion. Mitigations: Social boundary setting.",
    "category": "Vulnerability"
  },

  "CAT-2021-007": {
    "id": "CAT-2021-007",
    "name": "Low Conscientiousness",
    "description": "Low Conscientiousness reduces attention to detail. This vulnerability enables oversight exploitation. Mechanism: Lower self-discipline. Exploitation examples: Sloppy error inducement. Layer: 8 (individual). Related concepts: Carelessness. Mitigations: Checklist use.",
    "category": "Vulnerability"
  },

  "CAT-2021-006": {
    "id": "CAT-2021-006",
    "name": "Low Openness",
    "description": "Low Openness resists new ideas. This vulnerability can be exploited with incremental framing. Mechanism: Preference for familiar. Exploitation examples: Gradual radicalization. Layer: 8 (individual). Related concepts: Close-mindedness. Mitigations: Exposure training.",
    "category": "Vulnerability"
  },

  "CAT-2021-005": {
    "id": "CAT-2021-005",
    "name": "High Neuroticism",
    "description": "High Neuroticism increases emotional instability. This vulnerability amplifies fear/anxiety exploitation. Mechanism: Heightened stress response. Exploitation examples: Threat-based messaging. Layer: 8 (individual). Related concepts: Anxiety proneness. Mitigations: Stress management.",
    "category": "Vulnerability"
  },

  "CAT-2021-004": {
    "id": "CAT-2021-004",
    "name": "High Agreeableness",
    "description": "High Agreeableness increases trust and cooperation. This vulnerability enables excessive compliance. Mechanism: Conflict avoidance. Exploitation examples: Over-trusting scams. Layer: 8 (individual). Related concepts: Gullibility. Mitigations: Healthy skepticism.",
    "category": "Vulnerability"
  },

  "CAT-2021-003": {
    "id": "CAT-2021-003",
    "name": "High Extraversion",
    "description": "High Extraversion seeks stimulation and social interaction. This vulnerability enables novelty exploitation. Mechanism: Stimulation seeking. Exploitation examples: Exciting scam offers. Layer: 8 (individual). Related concepts: Sensation seeking. Mitigations: Risk assessment.",
    "category": "Vulnerability"
  },

  "CAT-2021-002": {
    "id": "CAT-2021-002",
    "name": "High Conscientiousness",
    "description": "High Conscientiousness leads to rigidity in habits. This vulnerability enables rule-based exploitation. Mechanism: Strong routine adherence. Exploitation examples: Predictable behavior attacks. Layer: 8 (individual). Related concepts: Obsessive patterns. Mitigations: Flexibility training.",
    "category": "Vulnerability"
  },
  
  "CAT-2021-001": {
    "id": "CAT-2021-001",
    "name": "High Openness",
    "description": "High Openness embraces novelty and alternative perspectives. This vulnerability enables over-acceptance of unverified ideas. Mechanism: Curiosity and tolerance. Exploitation examples: Conspiracy adoption. Layer: 8 (individual). Related concepts: Gullibility to novelty. Mitigations: Critical evaluation.",
    "category": "Vulnerability"
  }
}

DISARM_DESCRIPTIONS = {
    "DISARM-T0001": {"id": "DISARM-T0001", "name": "Plan Strategy", "description": "Develop overall strategy for disinformation campaign including objectives and target audiences. Mechanism: High-level planning.", "category": "TTP"},
    "DISARM-T0002": {"id": "DISARM-T0002", "name": "Facilitate State Propaganda", "description": "Organize citizens around pro-state messaging. Coordinate paid or volunteer groups to push state propaganda. Mechanism: State-coordinated message amplification.", "category": "TTP"},
    "DISARM-T0003": {"id": "DISARM-T0003", "name": "Leverage Existing Narratives", "description": "Use or adapt existing narrative themes where narratives form the bedrock of worldviews. Frame misinformation in context of prevailing narratives. Mechanism: Narrative alignment with existing beliefs.", "category": "TTP"},
    "DISARM-T0004": {"id": "DISARM-T0004", "name": "Develop Competing Narratives", "description": "Advance competing narratives connected to same issue. Construct alternatives centered on denial, deflection, dismissal, counter-charges. Mechanism: Narrative competition and confusion.", "category": "TTP"},
    "DISARM-T0007": {"id": "DISARM-T0007", "name": "Create Inauthentic Social Media Pages and Groups", "description": "Create key social engineering assets needed to amplify content, manipulate algorithms, fool public. Mechanism: False credibility through fake accounts.", "category": "TTP"},
    "DISARM-T0009": {"id": "DISARM-T0009", "name": "Create Fake Experts", "description": "Fabricate experts from whole cloth to lend credibility to stories. Mechanism: False authority exploitation.", "category": "TTP"},
    "DISARM-T0009.001": {"id": "DISARM-T0009.001", "name": "Utilize Academic/Pseudoscientific Justifications", "description": "Employ academic-sounding or pseudoscientific language to add false legitimacy. Mechanism: Scientific-seeming rhetoric.", "category": "TTP"},
    "DISARM-T0010": {"id": "DISARM-T0010", "name": "Cultivate Ignorant Agents", "description": "Cultivate propagandists whose goals are not fully comprehended. Use social media networks to amplify state disinformation. Mechanism: Unwitting agent manipulation.", "category": "TTP"},
    "DISARM-T0011": {"id": "DISARM-T0011", "name": "Compromise Legitimate Accounts", "description": "Hack or take over legitimate accounts to distribute misinformation or damaging content. Mechanism: Account hijacking.", "category": "TTP"},
    "DISARM-T0013": {"id": "DISARM-T0013", "name": "Create Inauthentic Websites", "description": "Create media assets to support inauthentic organizations or serve as sites to distribute malware. Mechanism: Fake web infrastructure.", "category": "TTP"},
    "DISARM-T0014": {"id": "DISARM-T0014", "name": "Prepare Fundraising Campaigns", "description": "Systematic effort to seek financial support for a cause while promoting operation messaging. Mechanism: Revenue generation with narrative amplification.", "category": "TTP"},
    "DISARM-T0014.001": {"id": "DISARM-T0014.001", "name": "Raise Funds from Malign Actors", "description": "Obtain contributions from foreign agents, cutouts, proxies, shell companies, dark money groups. Mechanism: Covert funding channels.", "category": "TTP"},
    "DISARM-T0014.002": {"id": "DISARM-T0014.002", "name": "Raise Funds from Ignorant Agents", "description": "Obtain funds through scams, donations intended for one purpose but used for another. Mechanism: Deceptive fundraising.", "category": "TTP"},
    "DISARM-T0015": {"id": "DISARM-T0015", "name": "Create Hashtags and Search Artifacts", "description": "Create hashtags to promote fabricated events, create perception of reality, publicize stories through trending. Mechanism: Hashtag-based reality construction.", "category": "TTP"},
    "DISARM-T0016": {"id": "DISARM-T0016", "name": "Create Clickbait", "description": "Create attention-grabbing headlines using outrage, doubt, or humor to drive traffic and engagement. Mechanism: Emotional manipulation for clicks.", "category": "TTP"},
    "DISARM-T0017": {"id": "DISARM-T0017", "name": "Conduct Fundraising", "description": "Systematic effort to seek financial support using online activities that promote operation pathways. Mechanism: Integrated fundraising and propaganda.", "category": "TTP"},
    "DISARM-T0017.001": {"id": "DISARM-T0017.001", "name": "Conduct Crowdfunding Campaigns", "description": "Use platforms like GoFundMe, GiveSendGo, Patreon for fundraising and message amplification. Mechanism: Crowdfunding platform exploitation.", "category": "TTP"},
    "DISARM-T0018": {"id": "DISARM-T0018", "name": "Purchase Targeted Advertisements", "description": "Create or fund advertisements targeted at specific populations. Mechanism: Paid targeting of audiences.", "category": "TTP"},
    "DISARM-T0019": {"id": "DISARM-T0019", "name": "Generate Information Pollution", "description": "Flood social channels to create aura of pervasiveness or consensus. Akin to astroturfing campaign. Mechanism: Information environment saturation.", "category": "TTP"},
    "DISARM-T0019.001": {"id": "DISARM-T0019.001", "name": "Create Fake Research", "description": "Create fake academic research aimed at hot-button social issues or pseudoscience. Mechanism: False scientific credibility.", "category": "TTP"},
    "DISARM-T0019.002": {"id": "DISARM-T0019.002", "name": "Hijack Hashtags", "description": "Use trending hashtags to promote topics substantially different from recent context. Mechanism: Hashtag co-option.", "category": "TTP"},
    "DISARM-T0020": {"id": "DISARM-T0020", "name": "Trial Content", "description": "Iteratively test incident performance through A/B testing of headlines, content engagement metrics. Mechanism: Content optimization through testing.", "category": "TTP"},
    "DISARM-T0022": {"id": "DISARM-T0022", "name": "Leverage Conspiracy Theory Narratives", "description": "Appeal to human desire for explanatory order by invoking powerful actors in pursuit of political goals. Mechanism: Conspiracy narrative exploitation.", "category": "TTP"},
    "DISARM-T0022.001": {"id": "DISARM-T0022.001", "name": "Amplify Existing Conspiracy Theory Narratives", "description": "Amplify existing conspiracy narratives that align with campaign goals to leverage existing communities. Mechanism: Conspiracy community activation.", "category": "TTP"},
    "DISARM-T0022.002": {"id": "DISARM-T0022.002", "name": "Develop Original Conspiracy Theory Narratives", "description": "Develop original conspiracy narratives for greater control and alignment with campaign goals. Mechanism: Custom conspiracy creation.", "category": "TTP"},
    "DISARM-T0023": {"id": "DISARM-T0023", "name": "Distort Facts", "description": "Change, twist, or exaggerate existing facts to construct narrative that differs from reality. Mechanism: Selective editing or fabrication.", "category": "TTP"},
    "DISARM-T0023.001": {"id": "DISARM-T0023.001", "name": "Reframe Context", "description": "Remove event from surrounding context to distort its intended meaning without denying it occurred. Mechanism: Context manipulation.", "category": "TTP"},
    "DISARM-T0023.002": {"id": "DISARM-T0023.002", "name": "Edit Open-Source Content", "description": "Edit collaborative blogs or encyclopedias to promote narratives on outlets with existing credibility. Mechanism: Collaborative platform manipulation.", "category": "TTP"},
    "DISARM-T0029": {"id": "DISARM-T0029", "name": "Online Polls", "description": "Create fake online polls or manipulate existing polls as data gathering tactic. Mechanism: Poll manipulation for targeting.", "category": "TTP"},
    "DISARM-T0039": {"id": "DISARM-T0039", "name": "Bait Legitimate Influencers", "description": "Target high-influence people with content engineered to appeal to their emotional drivers for unwitting amplification. Mechanism: Influencer manipulation for credibility.", "category": "TTP"},
    "DISARM-T0040": {"id": "DISARM-T0040", "name": "Demand Insurmountable Proof", "description": "Constantly escalate demands for proof to leverage asymmetry where truth-tellers are burdened with higher standards. Mechanism: Proof escalation exploitation.", "category": "TTP"},
    "DISARM-T0042": {"id": "DISARM-T0042", "name": "Seed Kernel of Truth", "description": "Wrap lies or altered context around truths to make messaging less likely to be dismissed. Mechanism: Truth-wrapped deception.", "category": "TTP"},
    "DISARM-T0043": {"id": "DISARM-T0043", "name": "Chat Apps", "description": "Direct messaging via chat app as delivery method, often automated, anonymous, viral, and ephemeral. Mechanism: Encrypted messaging exploitation.", "category": "TTP"},
    "DISARM-T0043.001": {"id": "DISARM-T0043.001", "name": "Use Encrypted Chat Apps", "description": "Utilize Signal, WhatsApp, Discord, Wire for covert communications. Mechanism: End-to-end encryption exploitation.", "category": "TTP"},
    "DISARM-T0043.002": {"id": "DISARM-T0043.002", "name": "Use Unencrypted Chats Apps", "description": "Utilize SMS and other unencrypted channels for message dissemination. Mechanism: Unencrypted mass messaging.", "category": "TTP"},
    "DISARM-T0044": {"id": "DISARM-T0044", "name": "Seed Distortions", "description": "Try wide variety of messages in early hours surrounding incident to give misleading account. Mechanism: Initial narrative flooding.", "category": "TTP"},
    "DISARM-T0045": {"id": "DISARM-T0045", "name": "Use Fake Experts", "description": "Deploy pseudo-experts as disposable assets to give credibility to misinformation. Mechanism: False credentials exploitation.", "category": "TTP"},
    "DISARM-T0046": {"id": "DISARM-T0046", "name": "Use Search Engine Optimization", "description": "Manipulate content engagement metrics to influence news search results and elevate propaganda headlines. Mechanism: Black-hat SEO techniques.", "category": "TTP"},
    "DISARM-T0047": {"id": "DISARM-T0047", "name": "Censor Social Media as Political Force", "description": "Use political influence or state power to stop critical social media comments through government-driven takedowns. Mechanism: State censorship.", "category": "TTP"},
    "DISARM-T0048": {"id": "DISARM-T0048", "name": "Harass", "description": "Threaten or harass believers of opposing narratives to discourage dissent through cyberbullying and doxing. Mechanism: Intimidation tactics.", "category": "TTP"},
    "DISARM-T0048.001": {"id": "DISARM-T0048.001", "name": "Boycott/Cancel Opponents", "description": "Exploit cancel culture by emphasizing adversary's problematic behavior to refrain support. Mechanism: Social cancellation.", "category": "TTP"},
    "DISARM-T0048.002": {"id": "DISARM-T0048.002", "name": "Harass People Based on Identities", "description": "Target individuals based on social identities like gender, race, religion, or roles like journalist. Mechanism: Identity-based harassment.", "category": "TTP"},
    "DISARM-T0048.003": {"id": "DISARM-T0048.003", "name": "Threaten to Dox", "description": "Threaten to release private information to discourage opposition from posting conflicting content. Mechanism: Doxing threat.", "category": "TTP"},
    "DISARM-T0048.004": {"id": "DISARM-T0048.004", "name": "Dox", "description": "Publicly release private information about individuals to encourage harassment or discourage dissent. Mechanism: Privacy violation.", "category": "TTP"},
    "DISARM-T0049": {"id": "DISARM-T0049", "name": "Flooding the Information Space", "description": "Flood social media channels with excessive content to control conversations and drown out opposing views. Mechanism: Volume-based suppression.", "category": "TTP"},
    "DISARM-T0049.001": {"id": "DISARM-T0049.001", "name": "Trolls Amplify and Manipulate", "description": "Use fake profiles operating across political spectrum to amplify narratives on divisive issues. Mechanism: Troll network coordination.", "category": "TTP"},
    "DISARM-T0049.002": {"id": "DISARM-T0049.002", "name": "Hijack Existing Hashtag", "description": "Take over existing hashtag to drive exposure to operation content. Mechanism: Hashtag hijacking.", "category": "TTP"},
    "DISARM-T0049.003": {"id": "DISARM-T0049.003", "name": "Bots Amplify via Automated Forwarding and Reposting", "description": "Use automated activity to amplify content above algorithm thresholds without human resources. Mechanism: Bot amplification.", "category": "TTP"},
    "DISARM-T0049.004": {"id": "DISARM-T0049.004", "name": "Utilize Spamoflauge", "description": "Disguise spam messages as legitimate by modifying letters, grammar, or encapsulating in protected files. Mechanism: Spam disguising.", "category": "TTP"},
    "DISARM-T0049.005": {"id": "DISARM-T0049.005", "name": "Conduct Swarming", "description": "Coordinate accounts to overwhelm information space with content around specific event or actor. Mechanism: Coordinated account swarming.", "category": "TTP"},
    "DISARM-T0049.006": {"id": "DISARM-T0049.006", "name": "Conduct Keyword Squatting", "description": "Create content around SEO terms to overwhelm search results and manipulate narrative. Mechanism: Search result manipulation.", "category": "TTP"},
    "DISARM-T0049.007": {"id": "DISARM-T0049.007", "name": "Inauthentic Sites Amplify News and Narratives", "description": "Use sites without masthead or bylines to cross-post and amplify narratives. Mechanism: Anonymous site amplification.", "category": "TTP"},
    "DISARM-T0057": {"id": "DISARM-T0057", "name": "Organize Events", "description": "Coordinate and promote real-world events like rallies and protests in support of incident narratives. Mechanism: Physical event organization.", "category": "TTP"},
    "DISARM-T0057.001": {"id": "DISARM-T0057.001", "name": "Pay for Physical Action", "description": "Pay individuals to act in physical realm to create situations supporting operation narratives. Mechanism: Paid physical actors.", "category": "TTP"},
    "DISARM-T0057.002": {"id": "DISARM-T0057.002", "name": "Conduct Symbolic Action", "description": "Conduct activities intended to advance narrative by signaling to audience through symbolic acts. Mechanism: Symbolic action creation.", "category": "TTP"},
    "DISARM-T0059": {"id": "DISARM-T0059", "name": "Play the Long Game", "description": "Plan messaging to grow organically over years or develop disconnected narratives that eventually combine. Mechanism: Long-term narrative development.", "category": "TTP"},
    "DISARM-T0060": {"id": "DISARM-T0060", "name": "Continue to Amplify", "description": "Continue narrative or message amplification after main incident work has finished. Mechanism: Sustained amplification.", "category": "TTP"},
    "DISARM-T0061": {"id": "DISARM-T0061", "name": "Sell Merchandise", "description": "Get message or narrative into physical space in offline world while making money. Mechanism: Merchandise-based propagation.", "category": "TTP"},
    "DISARM-T0065": {"id": "DISARM-T0065", "name": "Prepare Physical Broadcast Capabilities", "description": "Create or coopt broadcast capabilities like TV and radio. Mechanism: Traditional media control.", "category": "TTP"},
    "DISARM-T0066": {"id": "DISARM-T0066", "name": "Degrade Adversary", "description": "Plan to degrade adversary's image or ability to act using harmful information about their actions or reputation. Mechanism: Adversary degradation planning.", "category": "TTP"},
    "DISARM-T0068": {"id": "DISARM-T0068", "name": "Respond to Breaking News Event or Active Crisis", "description": "Exploit heightened media attention during breaking news where unclear facts increase manipulation vulnerability. Mechanism: Crisis exploitation.", "category": "TTP"},
    "DISARM-T0072": {"id": "DISARM-T0072", "name": "Segment Audiences", "description": "Create audience segmentations by political affiliation, geography, income, demographics, psychographics. Mechanism: Audience targeting.", "category": "TTP"},
    "DISARM-T0072.001": {"id": "DISARM-T0072.001", "name": "Geographic Segmentation", "description": "Target populations in specific geographic locations like region, state, or city. Mechanism: Location-based targeting.", "category": "TTP"},
    "DISARM-T0072.002": {"id": "DISARM-T0072.002", "name": "Demographic Segmentation", "description": "Target populations based on age, gender, and income characteristics. Mechanism: Demographic targeting.", "category": "TTP"},
    "DISARM-T0072.003": {"id": "DISARM-T0072.003", "name": "Economic Segmentation", "description": "Target populations based on income bracket, wealth, or financial division. Mechanism: Economic targeting.", "category": "TTP"},
    "DISARM-T0072.004": {"id": "DISARM-T0072.004", "name": "Psychographic Segmentation", "description": "Target populations based on values and decision-making processes using surveys or purchased data. Mechanism: Psychographic profiling.", "category": "TTP"},
    "DISARM-T0072.005": {"id": "DISARM-T0072.005", "name": "Political Segmentation", "description": "Target populations based on political affiliations to manipulate voting or change policy. Mechanism: Political targeting.", "category": "TTP"},
    "DISARM-T0073": {"id": "DISARM-T0073", "name": "Determine Target Audiences", "description": "Determine target audience segments who will receive campaign narratives to achieve strategic ends. Mechanism: Audience identification.", "category": "TTP"},
    "DISARM-T0074": {"id": "DISARM-T0074", "name": "Determine Strategic Ends", "description": "Determine campaign goals like geopolitical advantage, domestic political gain, financial gain, or policy change. Mechanism: Strategic objective setting.", "category": "TTP"},
    "DISARM-T0075": {"id": "DISARM-T0075", "name": "Dismiss", "description": "Push back against criticism by dismissing critics, arguing different standards or biased criticism. Mechanism: Criticism dismissal.", "category": "TTP"},
    "DISARM-T0075.001": {"id": "DISARM-T0075.001", "name": "Discredit Credible Sources", "description": "Delegitimize media landscape and degrade public trust in reporting by discrediting credible sources. Mechanism: Source credibility attack.", "category": "TTP"},
    "DISARM-T0076": {"id": "DISARM-T0076", "name": "Distort", "description": "Twist the narrative by taking information or artifacts and changing framing around them. Mechanism: Framing manipulation.", "category": "TTP"},
    "DISARM-T0077": {"id": "DISARM-T0077", "name": "Distract", "description": "Shift attention to different narrative or actor, like accusing critics of same activity. Mechanism: Attention redirection.", "category": "TTP"},
    "DISARM-T0078": {"id": "DISARM-T0078", "name": "Dismay", "description": "Threaten the critic or narrator of events like journalists or news outlets reporting on story. Mechanism: Threat-based intimidation.", "category": "TTP"},
    "DISARM-T0079": {"id": "DISARM-T0079", "name": "Divide", "description": "Create conflict between subgroups to widen divisions in community. Mechanism: Social division creation.", "category": "TTP"},
    "DISARM-T0080": {"id": "DISARM-T0080", "name": "Map Target Audience Information Environment", "description": "Analyze information space including social media analytics, web traffic, media surveys. Mechanism: Information environment analysis.", "category": "TTP"},
    "DISARM-T0080.001": {"id": "DISARM-T0080.001", "name": "Monitor Social Media Analytics", "description": "Use analytics to determine factors that increase content exposure on social media platforms. Mechanism: Platform analytics exploitation.", "category": "TTP"},
    "DISARM-T0080.002": {"id": "DISARM-T0080.002", "name": "Evaluate Media Surveys", "description": "Evaluate media surveys to determine what content appeals to target audience. Mechanism: Survey analysis.", "category": "TTP"},
    "DISARM-T0080.003": {"id": "DISARM-T0080.003", "name": "Identify Trending Topics/Hashtags", "description": "Identify trending hashtags on social media for later use in boosting operation content. Mechanism: Trend monitoring.", "category": "TTP"},
    "DISARM-T0080.004": {"id": "DISARM-T0080.004", "name": "Conduct Web Traffic Analysis", "description": "Conduct analysis to determine which search engines, keywords, websites gain most traction. Mechanism: Traffic pattern analysis.", "category": "TTP"},
    "DISARM-T0080.005": {"id": "DISARM-T0080.005", "name": "Assess Degree/Type of Media Access", "description": "Survey target audience's Internet availability and media freedom to determine content access. Mechanism: Access assessment.", "category": "TTP"},
    "DISARM-T0081": {"id": "DISARM-T0081", "name": "Identify Social and Technical Vulnerabilities", "description": "Determine weaknesses in target information environment including political issues, weak cybersecurity, data voids. Mechanism: Vulnerability identification.", "category": "TTP"},
    "DISARM-T0081.001": {"id": "DISARM-T0081.001", "name": "Find Echo Chambers", "description": "Find or plan to create areas where individuals only engage with people they agree with. Mechanism: Echo chamber identification.", "category": "TTP"},
    "DISARM-T0081.002": {"id": "DISARM-T0081.002", "name": "Identify Data Voids", "description": "Identify search terms with little or low-quality results for later exploitation during breaking news. Mechanism: Search void exploitation.", "category": "TTP"},
    "DISARM-T0081.003": {"id": "DISARM-T0081.003", "name": "Identify Existing Prejudices", "description": "Exploit existing racial, religious, demographic prejudices to further polarize target audience. Mechanism: Prejudice exploitation.", "category": "TTP"},
    "DISARM-T0081.004": {"id": "DISARM-T0081.004", "name": "Identify Existing Fissures", "description": "Identify existing divisions to pit populations against each other in divide-and-conquer approach. Mechanism: Social fissure exploitation.", "category": "TTP"},
    "DISARM-T0081.005": {"id": "DISARM-T0081.005", "name": "Identify Existing Conspiracy Narratives/Suspicions", "description": "Assess preexisting conspiracy theories or suspicions to identify narratives supporting objectives. Mechanism: Conspiracy narrative leveraging.", "category": "TTP"},
    "DISARM-T0081.006": {"id": "DISARM-T0081.006", "name": "Identify Wedge Issues", "description": "Exploit divisive political issues that split individuals along defined lines. Mechanism: Wedge issue exploitation.", "category": "TTP"},
    "DISARM-T0081.007": {"id": "DISARM-T0081.007", "name": "Identify Target Audience Adversaries", "description": "Identify or create real or imaginary adversary to center operation narratives against. Mechanism: Adversary creation.", "category": "TTP"},
    "DISARM-T0081.008": {"id": "DISARM-T0081.008", "name": "Identify Media System Vulnerabilities", "description": "Exploit weaknesses in target's media system including biases and existing distrust. Mechanism: Media system exploitation.", "category": "TTP"},
    "DISARM-T0082": {"id": "DISARM-T0082", "name": "Develop New Narratives", "description": "Develop new narratives for greater control in achieving goals when existing narratives don't align. Mechanism: Original narrative creation.", "category": "TTP"},
    "DISARM-T0083": {"id": "DISARM-T0083", "name": "Integrate Target Audience Vulnerabilities into Narrative", "description": "Exploit preexisting weaknesses, fears, enemies of target audience in operation narratives. Mechanism: Vulnerability integration.", "category": "TTP"},
    "DISARM-T0084": {"id": "DISARM-T0084", "name": "Reuse Existing Content", "description": "Recycle content from own previous operations or plagiarize from external operations. Mechanism: Content recycling.", "category": "TTP"},
    "DISARM-T0084.001": {"id": "DISARM-T0084.001", "name": "Use Copypasta", "description": "Use text copied and pasted multiple times across platforms, possibly edited as reposted. Mechanism: Viral text replication.", "category": "TTP"},
    "DISARM-T0084.002": {"id": "DISARM-T0084.002", "name": "Plagiarize Content", "description": "Take content from other sources without proper attribution. Mechanism: Content theft.", "category": "TTP"},
    "DISARM-T0084.003": {"id": "DISARM-T0084.003", "name": "Deceptively Labeled or Translated", "description": "Take authentic content and add deceptive labels or deceptively translate into other languages. Mechanism: Translation manipulation.", "category": "TTP"},
    "DISARM-T0084.004": {"id": "DISARM-T0084.004", "name": "Appropriate Content", "description": "Take content from other sources with proper attribution to leverage existing material. Mechanism: Content appropriation.", "category": "TTP"},
    "DISARM-T0085": {"id": "DISARM-T0085", "name": "Develop Text-based Content", "description": "Create and edit false or misleading text-based artifacts aligned with narratives. Mechanism: Text content creation.", "category": "TTP"},
    "DISARM-T0085.001": {"id": "DISARM-T0085.001", "name": "Develop AI-Generated Text", "description": "Use text-generating AI for autonomous content generation without human input. Mechanism: AI text generation.", "category": "TTP"},
    "DISARM-T0085.002": {"id": "DISARM-T0085.002", "name": "Develop False or Altered Documents", "description": "Create or modify documents to appear authentic for campaign goals. Mechanism: Document fabrication.", "category": "TTP"},
    "DISARM-T0085.003": {"id": "DISARM-T0085.003", "name": "Develop Inauthentic News Articles", "description": "Develop false or misleading news articles aligned to campaign goals or narratives. Mechanism: Fake news creation.", "category": "TTP"},
    "DISARM-T0086": {"id": "DISARM-T0086", "name": "Develop Image-based Content", "description": "Create and edit false or misleading visual artifacts aligned with narratives. Mechanism: Image content creation.", "category": "TTP"},
    "DISARM-T0086.001": {"id": "DISARM-T0086.001", "name": "Develop Memes", "description": "Create memes that pull together reference, commentary, image, narrative, emotion, and message. Mechanism: Meme creation.", "category": "TTP"},
    "DISARM-T0086.002": {"id": "DISARM-T0086.002", "name": "Develop AI-Generated Images (Deepfakes)", "description": "Use AI to create falsified photos depicting inauthentic situations. Mechanism: Image deepfake generation.", "category": "TTP"},
    "DISARM-T0086.003": {"id": "DISARM-T0086.003", "name": "Deceptively Edit Images (Cheap fakes)", "description": "Use less sophisticated measures to alter images and create false context. Mechanism: Image cheap fake.", "category": "TTP"},
    "DISARM-T0086.004": {"id": "DISARM-T0086.004", "name": "Aggregate Information into Evidence Collages", "description": "Create image files that aggregate positive evidence to support narratives. Mechanism: Evidence collage creation.", "category": "TTP"},
    "DISARM-T0087": {"id": "DISARM-T0087", "name": "Develop Video-based Content", "description": "Create and edit false or misleading video artifacts aligned with narratives. Mechanism: Video content creation.", "category": "TTP"},
    "DISARM-T0087.001": {"id": "DISARM-T0087.001", "name": "Develop AI-Generated Videos (Deepfakes)", "description": "Use AI to create falsified videos depicting inauthentic situations. Mechanism: Video deepfake generation.", "category": "TTP"},
    "DISARM-T0087.002": {"id": "DISARM-T0087.002", "name": "Deceptively Edit Video (Cheap fakes)", "description": "Use less sophisticated measures like slowing, speeding, cutting footage to create false context. Mechanism: Video cheap fake.", "category": "TTP"},
    "DISARM-T0088": {"id": "DISARM-T0088", "name": "Develop Audio-based Content", "description": "Create and edit false or misleading audio artifacts aligned with narratives. Mechanism: Audio content creation.", "category": "TTP"},
    "DISARM-T0088.001": {"id": "DISARM-T0088.001", "name": "Develop AI-Generated Audio (Deepfakes)", "description": "Use AI to create falsified audio depicting inauthentic situations through voice synthesis. Mechanism: Audio deepfake generation.", "category": "TTP"},
    "DISARM-T0088.002": {"id": "DISARM-T0088.002", "name": "Deceptively Edit Audio (Cheap fakes)", "description": "Use less sophisticated measures to alter audio and create false context. Mechanism: Audio cheap fake.", "category": "TTP"},
    "DISARM-T0089": {"id": "DISARM-T0089", "name": "Obtain Private Documents", "description": "Procure documents not publicly available by legal or illegal means for later leaking. Mechanism: Document acquisition.", "category": "TTP"},
    "DISARM-T0089.001": {"id": "DISARM-T0089.001", "name": "Obtain Authentic Documents", "description": "Procure authentic non-public documents by any means for later leaking. Mechanism: Document theft/acquisition.", "category": "TTP"},
    "DISARM-T0089.002": {"id": "DISARM-T0089.002", "name": "Create Inauthentic Documents", "description": "Create inauthentic documents intended to appear authentic for later leaking. Mechanism: Fake document creation.", "category": "TTP"},
    "DISARM-T0089.003": {"id": "DISARM-T0089.003", "name": "Alter Authentic Documents", "description": "Alter authentic documents to achieve campaign goals while appearing authentic. Mechanism: Document alteration.", "category": "TTP"},
    "DISARM-T0090": {"id": "DISARM-T0090", "name": "Create Inauthentic Accounts", "description": "Create bot, cyborg, sockpuppet, and anonymous accounts for content distribution. Mechanism: Fake account creation.", "category": "TTP"},
    "DISARM-T0090.001": {"id": "DISARM-T0090.001", "name": "Create Anonymous Accounts", "description": "Create accounts that access network resources without username or password. Mechanism: Anonymous account creation.", "category": "TTP"},
    "DISARM-T0090.002": {"id": "DISARM-T0090.002", "name": "Create Cyborg Accounts", "description": "Create partly manned, partly automated accounts with periodic human control. Mechanism: Hybrid account creation.", "category": "TTP"},
    "DISARM-T0090.003": {"id": "DISARM-T0090.003", "name": "Create Bot Accounts", "description": "Create autonomous accounts that imitate human behavior using AI and big data. Mechanism: Bot account creation.", "category": "TTP"},
    "DISARM-T0090.004": {"id": "DISARM-T0090.004", "name": "Create Sockpuppet Accounts", "description": "Create falsified accounts to promote material or attack critics online. Mechanism: Sockpuppet creation.", "category": "TTP"},
    "DISARM-T0091": {"id": "DISARM-T0091", "name": "Recruit Malign Actors", "description": "Recruit bad actors by paying, recruiting, or exerting control including trolls, partisans, contractors. Mechanism: Actor recruitment.", "category": "TTP"},
    "DISARM-T0091.001": {"id": "DISARM-T0091.001", "name": "Recruit Contractors", "description": "Recruit paid contractors to support the campaign. Mechanism: Contractor hiring.", "category": "TTP"},
    "DISARM-T0091.002": {"id": "DISARM-T0091.002", "name": "Recruit Partisans", "description": "Recruit ideologically-aligned individuals to support the campaign. Mechanism: Partisan recruitment.", "category": "TTP"},
    "DISARM-T0091.003": {"id": "DISARM-T0091.003", "name": "Enlist Troll Accounts", "description": "Hire trolls or human operators of fake accounts to provoke others and discredit opposition. Mechanism: Troll hiring.", "category": "TTP"},
    "DISARM-T0092": {"id": "DISARM-T0092", "name": "Build Network", "description": "Build own network creating links between accounts to amplify and promote narratives. Mechanism: Network construction.", "category": "TTP"},
    "DISARM-T0092.001": {"id": "DISARM-T0092.001", "name": "Create Organizations", "description": "Establish organizations with legitimate or falsified hierarchies for operational structure. Mechanism: Organizational creation.", "category": "TTP"},
    "DISARM-T0092.002": {"id": "DISARM-T0092.002", "name": "Use Follow Trains", "description": "Use groups who follow each other on social media to grow following. Mechanism: Follow-for-follow exploitation.", "category": "TTP"},
    "DISARM-T0092.003": {"id": "DISARM-T0092.003", "name": "Create Community or Sub-group", "description": "Create new community or sub-group when existing ones don't meet campaign goals. Mechanism: Community creation.", "category": "TTP"},
    "DISARM-T0093": {"id": "DISARM-T0093", "name": "Acquire/Recruit Network", "description": "Acquire existing network by paying, recruiting, or exerting control over leaders. Mechanism: Network acquisition.", "category": "TTP"},
    "DISARM-T0093.001": {"id": "DISARM-T0093.001", "name": "Fund Proxies", "description": "Fund external entities or users with existing sympathies toward operation narratives. Mechanism: Proxy funding.", "category": "TTP"},
    "DISARM-T0093.002": {"id": "DISARM-T0093.002", "name": "Acquire Botnets", "description": "Acquire group of bots that can function in coordination with each other. Mechanism: Botnet acquisition.", "category": "TTP"},
    "DISARM-T0094": {"id": "DISARM-T0094", "name": "Infiltrate Existing Networks", "description": "Deceptively insert social assets into existing networks as members to influence the network. Mechanism: Network infiltration.", "category": "TTP"},
    "DISARM-T0094.001": {"id": "DISARM-T0094.001", "name": "Identify Susceptible Targets in Networks", "description": "Identify individuals and groups susceptible to being co-opted or influenced. Mechanism: Target identification.", "category": "TTP"},
    "DISARM-T0094.002": {"id": "DISARM-T0094.002", "name": "Utilize Butterfly Attacks", "description": "Pretend to be members of social group to insert controversial statements and discredit movements. Mechanism: Group impersonation.", "category": "TTP"},
    "DISARM-T0095": {"id": "DISARM-T0095", "name": "Develop Owned Media Assets", "description": "Create agency or organization to create, develop, and host content through owned platforms. Mechanism: Media asset development.", "category": "TTP"},
    "DISARM-T0096": {"id": "DISARM-T0096", "name": "Leverage Content Farms", "description": "Use services of large-scale content providers for creating and amplifying campaign artifacts at scale. Mechanism: Content farm utilization.", "category": "TTP"},
    "DISARM-T0096.001": {"id": "DISARM-T0096.001", "name": "Create Content Farms", "description": "Create organization for creating and amplifying campaign artifacts at scale. Mechanism: Content farm creation.", "category": "TTP"},
    "DISARM-T0096.002": {"id": "DISARM-T0096.002", "name": "Outsource Content Creation to External Organizations", "description": "Outsource to external companies to avoid attribution, increase creation rate, or improve quality. Mechanism: Content outsourcing.", "category": "TTP"},
    "DISARM-T0097": {"id": "DISARM-T0097", "name": "Create Personas", "description": "Create fake people with accounts across platforms ranging from simple names to fully backstopped identities. Mechanism: Persona fabrication.", "category": "TTP"},
    "DISARM-T0097.001": {"id": "DISARM-T0097.001", "name": "Backstop Personas", "description": "Create additional assets, cover, fake relationships, documents to establish credibility. Mechanism: Persona backstopping.", "category": "TTP"},
    "DISARM-T0098": {"id": "DISARM-T0098", "name": "Establish Inauthentic News Sites", "description": "Create or leverage imposter news sites with superficial markers of authenticity. Mechanism: Fake news site establishment.", "category": "TTP"},
    "DISARM-T0098.001": {"id": "DISARM-T0098.001", "name": "Create Inauthentic News Sites", "description": "Build new fake news sites from scratch with false legitimacy markers. Mechanism: News site creation.", "category": "TTP"},
    "DISARM-T0098.002": {"id": "DISARM-T0098.002", "name": "Leverage Existing Inauthentic News Sites", "description": "Use already established fake news sites for content distribution. Mechanism: Existing site leverage.", "category": "TTP"},
    "DISARM-T0099": {"id": "DISARM-T0099", "name": "Prepare Assets Impersonating Legitimate Entities", "description": "Prepare assets impersonating news outlets, public figures, organizations using typosquatting and spoofing. Mechanism: Entity impersonation.", "category": "TTP"},
    "DISARM-T0099.001": {"id": "DISARM-T0099.001", "name": "Astroturfing", "description": "Disguise operation as grassroots movement to increase appearance of popular support. Mechanism: Fake grassroots creation.", "category": "TTP"},
    "DISARM-T0099.002": {"id": "DISARM-T0099.002", "name": "Spoof/Parody Account/Site", "description": "Impersonate legitimate entities through spoofed or parody accounts and sites. Mechanism: Spoofing/parody.", "category": "TTP"},
    "DISARM-T0100": {"id": "DISARM-T0100", "name": "Co-opt Trusted Sources", "description": "Infiltrate or repurpose trusted sources to reach target audience through existing reliable networks. Mechanism: Source co-option.", "category": "TTP"},
    "DISARM-T0100.001": {"id": "DISARM-T0100.001", "name": "Co-Opt Trusted Individuals", "description": "Infiltrate or repurpose trusted individuals to leverage their credibility. Mechanism: Individual co-option.", "category": "TTP"},
    "DISARM-T0100.002": {"id": "DISARM-T0100.002", "name": "Co-Opt Grassroots Groups", "description": "Infiltrate or repurpose grassroots groups to leverage their authenticity. Mechanism: Grassroots co-option.", "category": "TTP"},
    "DISARM-T0100.003": {"id": "DISARM-T0100.003", "name": "Co-opt Influencers", "description": "Infiltrate or repurpose influencers to leverage their reach and credibility. Mechanism: Influencer co-option.", "category": "TTP"},
    "DISARM-T0101": {"id": "DISARM-T0101", "name": "Create Localized Content", "description": "Create content that appeals to specific communities using local language and dialects. Mechanism: Content localization.", "category": "TTP"},
    "DISARM-T0102": {"id": "DISARM-T0102", "name": "Leverage Echo Chambers/Filter Bubbles", "description": "Create or use isolated internet areas where individuals engage with like-minded others. Mechanism: Echo chamber exploitation.", "category": "TTP"},
    "DISARM-T0102.001": {"id": "DISARM-T0102.001", "name": "Use Existing Echo Chambers/Filter Bubbles", "description": "Leverage already existing echo chambers and filter bubbles for content distribution. Mechanism: Existing echo chamber use.", "category": "TTP"},
    "DISARM-T0102.002": {"id": "DISARM-T0102.002", "name": "Create Echo Chambers/Filter Bubbles", "description": "Build new echo chambers and filter bubbles to isolate target audiences. Mechanism: Echo chamber creation.", "category": "TTP"},
    "DISARM-T0102.003": {"id": "DISARM-T0102.003", "name": "Exploit Data Voids", "description": "Exploit search terms with little or manipulative results to proliferate false information quickly. Mechanism: Data void exploitation.", "category": "TTP"},
    "DISARM-T0103": {"id": "DISARM-T0103", "name": "Livestream", "description": "Use online broadcast capability for real-time communication to closed or open networks. Mechanism: Real-time broadcasting.", "category": "TTP"},
    "DISARM-T0103.001": {"id": "DISARM-T0103.001", "name": "Video Livestream", "description": "Use online video broadcast for real-time visual communication. Mechanism: Video streaming.", "category": "TTP"},
    "DISARM-T0103.002": {"id": "DISARM-T0103.002", "name": "Audio Livestream", "description": "Use online audio broadcast for real-time audio communication. Mechanism: Audio streaming.", "category": "TTP"},
    "DISARM-T0104": {"id": "DISARM-T0104", "name": "Social Networks", "description": "Use interactive digital channels that facilitate creation and sharing of information. Mechanism: Social network exploitation.", "category": "TTP"},
    "DISARM-T0104.001": {"id": "DISARM-T0104.001", "name": "Mainstream Social Networks", "description": "Use platforms like Facebook, Twitter, LinkedIn for content distribution. Mechanism: Mainstream platform use.", "category": "TTP"},
    "DISARM-T0104.002": {"id": "DISARM-T0104.002", "name": "Dating Apps", "description": "Use dating applications for targeted relationship-based manipulation. Mechanism: Dating app exploitation.", "category": "TTP"},
    "DISARM-T0104.003": {"id": "DISARM-T0104.003", "name": "Private/Closed Social Networks", "description": "Use private or closed networks for coordinated activity. Mechanism: Private network use.", "category": "TTP"},
    "DISARM-T0104.004": {"id": "DISARM-T0104.004", "name": "Interest-Based Networks", "description": "Use smaller and niche networks including Gettr, Truth Social, Parler. Mechanism: Niche platform use.", "category": "TTP"},
    "DISARM-T0104.005": {"id": "DISARM-T0104.005", "name": "Use Hashtags", "description": "Use dedicated, existing hashtag for the campaign or incident. Mechanism: Hashtag utilization.", "category": "TTP"},
    "DISARM-T0104.006": {"id": "DISARM-T0104.006", "name": "Create Dedicated Hashtag", "description": "Create campaign or incident specific hashtag for coordination. Mechanism: Hashtag creation.", "category": "TTP"},
    "DISARM-T0105": {"id": "DISARM-T0105", "name": "Media Sharing Networks", "description": "Use services whose primary function is hosting and sharing specific forms of media. Mechanism: Media platform exploitation.", "category": "TTP"},
    "DISARM-T0105.001": {"id": "DISARM-T0105.001", "name": "Photo Sharing", "description": "Use platforms like Instagram, Snapchat, Flickr for image-based content. Mechanism: Photo platform use.", "category": "TTP"},
    "DISARM-T0105.002": {"id": "DISARM-T0105.002", "name": "Video Sharing", "description": "Use platforms like Youtube, TikTok, ShareChat, Rumble for video content. Mechanism: Video platform use.", "category": "TTP"},
    "DISARM-T0105.003": {"id": "DISARM-T0105.003", "name": "Audio Sharing", "description": "Use podcasting apps and Soundcloud for audio content distribution. Mechanism: Audio platform use.", "category": "TTP"},
    "DISARM-T0106": {"id": "DISARM-T0106", "name": "Discussion Forums", "description": "Use platforms for finding, discussing, and sharing information and opinions. Mechanism: Forum exploitation.", "category": "TTP"},
    "DISARM-T0106.001": {"id": "DISARM-T0106.001", "name": "Anonymous Message Boards", "description": "Use anonymous boards like the Chans for unattributed content distribution. Mechanism: Anonymous board use.", "category": "TTP"},
    "DISARM-T0107": {"id": "DISARM-T0107", "name": "Bookmarking and Content Curation", "description": "Use platforms for searching, sharing, and curating content like Pinterest, Flipboard. Mechanism: Curation platform use.", "category": "TTP"},
    "DISARM-T0108": {"id": "DISARM-T0108", "name": "Blogging and Publishing Networks", "description": "Use WordPress, Blogger, Weebly, Tumblr, Medium for long-form content. Mechanism: Blogging platform use.", "category": "TTP"},
    "DISARM-T0109": {"id": "DISARM-T0109", "name": "Consumer Review Networks", "description": "Use platforms like Yelp, TripAdvisor for review-based influence. Mechanism: Review platform manipulation.", "category": "TTP"},
    "DISARM-T0110": {"id": "DISARM-T0110", "name": "Formal Diplomatic Channels", "description": "Leverage traditional diplomatic channels to communicate with foreign governments. Mechanism: Diplomatic channel use.", "category": "TTP"},
    "DISARM-T0111": {"id": "DISARM-T0111", "name": "Traditional Media", "description": "Use TV, newspaper, radio for content distribution. Mechanism: Traditional media exploitation.", "category": "TTP"},
    "DISARM-T0111.001": {"id": "DISARM-T0111.001", "name": "TV", "description": "Use television broadcasting for wide-reach content distribution. Mechanism: TV broadcasting.", "category": "TTP"},
    "DISARM-T0111.002": {"id": "DISARM-T0111.002", "name": "Newspaper", "description": "Use print and online newspapers for content distribution. Mechanism: Print media use.", "category": "TTP"},
    "DISARM-T0111.003": {"id": "DISARM-T0111.003", "name": "Radio", "description": "Use radio broadcasting for audio-based content distribution. Mechanism: Radio broadcasting.", "category": "TTP"},
    "DISARM-T0112": {"id": "DISARM-T0112", "name": "Email", "description": "Deliver content and narratives via email including list management or targeted messaging. Mechanism: Email distribution.", "category": "TTP"},
    "DISARM-T0113": {"id": "DISARM-T0113", "name": "Employ Commercial Analytic Firms", "description": "Use commercial firms to facilitate external collection on target audience and tailor content. Mechanism: Analytics outsourcing.", "category": "TTP"},
    "DISARM-T0114": {"id": "DISARM-T0114", "name": "Deliver Ads", "description": "Deliver content via any form of paid media or advertising. Mechanism: Paid advertising.", "category": "TTP"},
    "DISARM-T0114.001": {"id": "DISARM-T0114.001", "name": "Social Media Ads", "description": "Purchase advertisements on social media platforms for targeted delivery. Mechanism: Social media advertising.", "category": "TTP"},
    "DISARM-T0114.002": {"id": "DISARM-T0114.002", "name": "Traditional Media Ads", "description": "Purchase advertisements on TV, Radio, Newspaper, billboards. Mechanism: Traditional advertising.", "category": "TTP"},
    "DISARM-T0115": {"id": "DISARM-T0115", "name": "Post Content", "description": "Deliver content by posting via owned media assets that operator controls. Mechanism: Direct posting.", "category": "TTP"},
    "DISARM-T0115.001": {"id": "DISARM-T0115.001", "name": "Share Memes", "description": "Post and share memes as powerful tools for message propagation. Mechanism: Meme sharing.", "category": "TTP"},
    "DISARM-T0115.002": {"id": "DISARM-T0115.002", "name": "Post Violative Content to Provoke Takedown and Backlash", "description": "Post content that violates platform rules to generate controversy when removed. Mechanism: Strategic violation.", "category": "TTP"},
    "DISARM-T0115.003": {"id": "DISARM-T0115.003", "name": "One-Way Direct Posting", "description": "Post via one-way messaging where recipients cannot directly respond to avoid debate. Mechanism: Unidirectional posting.", "category": "TTP"},
    "DISARM-T0116": {"id": "DISARM-T0116", "name": "Comment or Reply on Content", "description": "Deliver content by replying or commenting via owned media assets. Mechanism: Comment-based delivery.", "category": "TTP"},
    "DISARM-T0116.001": {"id": "DISARM-T0116.001", "name": "Post Inauthentic Social Media Comment", "description": "Use paid commenters, astroturfers, chat bots to influence online conversations. Mechanism: Inauthentic commenting.", "category": "TTP"},
    "DISARM-T0117": {"id": "DISARM-T0117", "name": "Attract Traditional Media", "description": "Deliver content by attracting attention of traditional media as earned media. Mechanism: Media attention attraction.", "category": "TTP"},
    "DISARM-T0118": {"id": "DISARM-T0118", "name": "Amplify Existing Narrative", "description": "Amplify existing narratives that align with operation narratives to support objectives. Mechanism: Narrative amplification.", "category": "TTP"},
    "DISARM-T0119": {"id": "DISARM-T0119", "name": "Cross-Posting", "description": "Post same message to multiple discussions, platforms, or accounts simultaneously. Mechanism: Multi-platform posting.", "category": "TTP"},
    "DISARM-T0119.001": {"id": "DISARM-T0119.001", "name": "Post Across Groups", "description": "Post content across groups to spread narratives to new communities. Mechanism: Cross-group posting.", "category": "TTP"},
    "DISARM-T0119.002": {"id": "DISARM-T0119.002", "name": "Post Across Platform", "description": "Post content across platforms to reach new audiences and remove opposition. Mechanism: Cross-platform posting.", "category": "TTP"},
    "DISARM-T0119.003": {"id": "DISARM-T0119.003", "name": "Post Across Disciplines", "description": "Post content across different subject areas and disciplines. Mechanism: Cross-discipline posting.", "category": "TTP"},
    "DISARM-T0120": {"id": "DISARM-T0120", "name": "Incentivize Sharing", "description": "Encourage users to share content themselves to reduce operation posting burden. Mechanism: Sharing incentives.", "category": "TTP"},
    "DISARM-T0120.001": {"id": "DISARM-T0120.001", "name": "Use Affiliate Marketing Programs", "description": "Use affiliate programs to incentivize content sharing through revenue. Mechanism: Affiliate incentives.", "category": "TTP"},
    "DISARM-T0120.002": {"id": "DISARM-T0120.002", "name": "Use Contests and Prizes", "description": "Offer contests and prizes to encourage content sharing and engagement. Mechanism: Prize-based incentives.", "category": "TTP"},
    "DISARM-T0121": {"id": "DISARM-T0121", "name": "Manipulate Platform Algorithm", "description": "Conduct activity targeting platform's algorithm to increase exposure or avoid removal. Mechanism: Algorithm exploitation.", "category": "TTP"},
    "DISARM-T0121.001": {"id": "DISARM-T0121.001", "name": "Bypass Content Blocking", "description": "Circumvent network security measures using VPNs, altered IPs, encryption to avoid blocking. Mechanism: Block circumvention.", "category": "TTP"},
    "DISARM-T0122": {"id": "DISARM-T0122", "name": "Direct Users to Alternative Platforms", "description": "Encourage users to move to alternate platforms to diversify information channels. Mechanism: Platform migration.", "category": "TTP"},
    "DISARM-T0123": {"id": "DISARM-T0123", "name": "Control Information Environment through Offensive Cyberspace Operations", "description": "Use cyber tools to alter content trajectory to prioritize operation messaging or block opposition. Mechanism: Cyber operations.", "category": "TTP"},
    "DISARM-T0123.001": {"id": "DISARM-T0123.001", "name": "Delete Opposing Content", "description": "Remove content that conflicts with operational narratives from selected platforms. Mechanism: Content deletion.", "category": "TTP"},
    "DISARM-T0123.002": {"id": "DISARM-T0123.002", "name": "Block Content", "description": "Restrict internet access or render certain areas inaccessible to limit opposition. Mechanism: Access blocking.", "category": "TTP"},
    "DISARM-T0123.003": {"id": "DISARM-T0123.003", "name": "Destroy Information Generation Capabilities", "description": "Limit, degrade, or incapacitate actor's ability to generate conflicting information. Mechanism: Capability destruction.", "category": "TTP"},
    "DISARM-T0123.004": {"id": "DISARM-T0123.004", "name": "Conduct Server Redirect", "description": "Automatically forward users from one URL to another without their knowledge. Mechanism: URL redirection.", "category": "TTP"},
    "DISARM-T0124": {"id": "DISARM-T0124", "name": "Suppress Opposition", "description": "Exploit platform content moderation tools to suppress opposition through reporting and goading. Mechanism: Opposition suppression.", "category": "TTP"},
    "DISARM-T0124.001": {"id": "DISARM-T0124.001", "name": "Report Non-Violative Opposing Content", "description": "Report opposing content as violating policies to trigger removal by platforms. Mechanism: False reporting.", "category": "TTP"},
    "DISARM-T0124.002": {"id": "DISARM-T0124.002", "name": "Goad People into Harmful Action", "description": "Goad people into actions that violate terms of service to get their content removed. Mechanism: Behavior provocation.", "category": "TTP"},
    "DISARM-T0124.003": {"id": "DISARM-T0124.003", "name": "Exploit Platform TOS/Content Moderation", "description": "Exploit weaknesses in platform policies and moderation to benefit operation. Mechanism: Policy exploitation.", "category": "TTP"},
    "DISARM-T0125": {"id": "DISARM-T0125", "name": "Platform Filtering", "description": "Decontextualize information as claims cross platforms to distort meaning. Mechanism: Cross-platform decontextualization.", "category": "TTP"},
    "DISARM-T0126": {"id": "DISARM-T0126", "name": "Encourage Attendance at Events", "description": "Encourage attendance at existing real world events to amplify impact. Mechanism: Event attendance promotion.", "category": "TTP"},
    "DISARM-T0126.001": {"id": "DISARM-T0126.001", "name": "Call to Action to Attend", "description": "Issue calls to action encouraging target audience to attend events. Mechanism: Attendance call.", "category": "TTP"},
    "DISARM-T0126.002": {"id": "DISARM-T0126.002", "name": "Facilitate Logistics or Support for Attendance", "description": "Provide travel, food, housing support to facilitate event attendance. Mechanism: Logistical support.", "category": "TTP"},
    "DISARM-T0127": {"id": "DISARM-T0127", "name": "Physical Violence", "description": "Conduct or encourage physical violence to discourage opponents or draw attention through shock value. Mechanism: Violence as tactic.", "category": "TTP"},
    "DISARM-T0127.001": {"id": "DISARM-T0127.001", "name": "Conduct Physical Violence", "description": "Directly conduct physical violence to achieve campaign goals. Mechanism: Direct violence.", "category": "TTP"},
    "DISARM-T0127.002": {"id": "DISARM-T0127.002", "name": "Encourage Physical Violence", "description": "Encourage others to engage in physical violence to achieve campaign goals. Mechanism: Violence incitement.", "category": "TTP"},
    "DISARM-T0128": {"id": "DISARM-T0128", "name": "Conceal People", "description": "Conceal identity or provenance of campaign accounts and people assets to avoid takedown. Mechanism: Identity concealment.", "category": "TTP"},
    "DISARM-T0128.001": {"id": "DISARM-T0128.001", "name": "Use Pseudonyms", "description": "Use fake names to mask identity of operation accounts and publish anonymous content. Mechanism: Pseudonym use.", "category": "TTP"},
    "DISARM-T0128.002": {"id": "DISARM-T0128.002", "name": "Conceal Network Identity", "description": "Hide the existence of influence operation's network completely. Mechanism: Network concealment.", "category": "TTP"},
    "DISARM-T0128.003": {"id": "DISARM-T0128.003", "name": "Distance Reputable Individuals from Operation", "description": "Have enlisted individuals actively disengage from operation activities and messaging. Mechanism: Individual distancing.", "category": "TTP"},
    "DISARM-T0128.004": {"id": "DISARM-T0128.004", "name": "Launder Accounts", "description": "Acquire control of previously legitimate accounts from third parties through sale or exchange. Mechanism: Account laundering.", "category": "TTP"},
    "DISARM-T0128.005": {"id": "DISARM-T0128.005", "name": "Change Names of Accounts", "description": "Change names of existing accounts to avoid detection or fit operational narratives. Mechanism: Name changing.", "category": "TTP"}
}

# Precompute embeddings for both registries at module load
print("Precomputing TTP embeddings with nomic-embed-text-v1.5...")

# CAT Registry (existing)
cat_texts = [entry["description"] for entry in TTP_DESCRIPTIONS.values()]
cat_embeddings = embed_texts(cat_texts)
CAT_EMBEDDINGS = {entry["id"]: emb for entry, emb in zip(TTP_DESCRIPTIONS.values(), cat_embeddings)}

# DISARM Registry (new)
disarm_texts = [entry["description"] for entry in DISARM_DESCRIPTIONS.values()]
disarm_embeddings = embed_texts(disarm_texts)
DISARM_EMBEDDINGS = {entry["id"]: emb for entry, emb in zip(DISARM_DESCRIPTIONS.values(), disarm_embeddings)}

# Combined total for status
total_ttps = len(CAT_EMBEDDINGS) + len(DISARM_EMBEDDINGS)

print(f"Embeddings ready:")
print(f"  - CAT registry: {len(CAT_EMBEDDINGS)} TTPs")
print(f"  - DISARM registry: {len(DISARM_EMBEDDINGS)} TTPs")
print(f"  - Total: {total_ttps} TTPs loaded")

def identify_ttp_dual(
    narrative_text: str,
    top_k_per_registry: int = 5,
    min_confidence: float = 0.45,
    prioritize_cognitive: bool = True
) -> List[Dict[str, Any]]:
    """
    Dual-registry TTP identification: CAT + DISARM
    - Ensures at least top_k_per_registry from each registry (when available)
    - Falls back to best overall if one registry is short
    - Single unified list for radar (max 10 total)
    """
    if not narrative_text.strip():
        return []

    narrative_emb = embed_texts([narrative_text])[0]

    # === CAT Registry Matching ===
    cat_matches = []
    for ttp_id, ttp_emb in CAT_EMBEDDINGS.items():
        sim = float(np.dot(narrative_emb, ttp_emb))

        entry = TTP_DESCRIPTIONS[ttp_id]

        if prioritize_cognitive:
            category = entry.get("category", "").lower()
            if category in ["vulnerability", "exploit"]:
                sim *= 1
            elif "layer" in entry["description"].lower() and "7" in entry["description"]:
                sim *= 0.8

            tech_keywords = [
            "ultrasonic", "model inversion", "backdoor", "prompt injection",
            "coercion", "torture", "extract keys", "rubber-hose", "cryptanalysis",
            "encryption", "blockchain", "quantum", "cryptography", "malware", "trojan"
            ]
            if any(kw in entry["description"].lower() for kw in tech_keywords):
                if not any(kw in narrative_text.lower() for kw in tech_keywords):
                    sim *= 0.3  # Very strong penalty (was 0.6)

        if sim >= min_confidence:
            cat_matches.append({
                "id": ttp_id,
                "name": entry["name"],
                "confidence": round(sim, 3),
                "source": "CAT"
            })

    cat_matches.sort(key=lambda x: x["confidence"], reverse=True)

    # === DISARM Registry Matching ===
    disarm_matches = []
    for ttp_id, ttp_emb in DISARM_EMBEDDINGS.items():
        sim = float(np.dot(narrative_emb, ttp_emb))

        entry = DISARM_DESCRIPTIONS[ttp_id]

        # Optional: lighter penalty for DISARM (more operational focus)
        tech_keywords_disarm = ["deepfake", "bot", "hashtag", "inauthentic"]
        if any(kw in entry["description"].lower() for kw in tech_keywords_disarm):
            if not any(kw in narrative_text.lower() for kw in tech_keywords_disarm):
                sim *= 0.7  # Slightly less aggressive than CAT

        if sim >= min_confidence:
            disarm_matches.append({
                "id": ttp_id,
                "name": entry["name"],
                "confidence": round(sim, 3),
                "source": "DISARM"
            })

    disarm_matches.sort(key=lambda x: x["confidence"], reverse=True)

    # === Enforce Minimum 5 from Each ===
    final_results = []

    # Take top from CAT
    final_results.extend(cat_matches[:top_k_per_registry])
    # Take top from DISARM
    final_results.extend(disarm_matches[:top_k_per_registry])

    # Total target
    total_target = top_k_per_registry * 2  # 10

    # If short, fill with next best from either
    if len(final_results) < total_target:
        remaining_needed = total_target - len(final_results)
        extra_cat = cat_matches[top_k_per_registry:top_k_per_registry + remaining_needed]
        extra_disarm = disarm_matches[top_k_per_registry:top_k_per_registry + remaining_needed]
        extras = extra_cat + extra_disarm
        extras.sort(key=lambda x: x["confidence"], reverse=True)
        final_results.extend(extras[:remaining_needed])

    # Final sort by confidence
    final_results.sort(key=lambda x: x["confidence"], reverse=True)

    # Cap at total_target
    return final_results[:total_target]