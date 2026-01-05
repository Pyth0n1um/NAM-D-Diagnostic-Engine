from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# UPGRADED EMBEDDING MODEL
# -----------------------------
# multi-qa-mpnet-base-dot-v1: Optimized for dot-product similarity
# Excellent for semantic clustering of technical/abstract concepts like TTPs
# Loads once at import time (~3-5 seconds on first run)
MODEL = SentenceTransformer(
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
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

# Define TTP descriptions (expanded with CAT vulnerabilities)
TTP_DESCRIPTIONS = {
"CAT-2025-001": "Vigilance refers to the sustained attention and alertness required to detect and respond to threats or anomalies in one's environment. This vulnerability arises from the finite nature of human attentional resources, leading to fatigue, habituation, or distraction over time. In cognitive security contexts, attackers exploit vigilance degradation through prolonged exposure to low-level stimuli or timing attacks during periods of low arousal (e.g., late night or repetitive tasks). Mechanism: Attentional resources deplete via sustained effort, causing decreased sensitivity to signals (signal detection theory). Exploitation examples: DDoS attacks on mental capacity through information overload, or phishing during end-of-day fatigue. Layer: 8 (individual cognition). Related concepts: Attention bias, divided attention. Mitigations: Automated monitoring tools or rotation schedules to maintain vigilance levels.",
"CAT-2024-007": "Need to Correct is the compulsive drive to rectify perceived errors, misinformation, or injustices, often triggering immediate action without full verification. This vulnerability stems from cognitive dissonance and the human desire for accuracy and fairness, making individuals susceptible to baiting tactics. Attackers exploit it by planting deliberate errors or provocations to elicit responses that reveal information or consume resources. Mechanism: Activation of error-detection circuits in the brain (anterior cingulate cortex), leading to impulsive correction behaviors. Exploitation examples: Trolls posting false facts on social media to draw out corrections from experts, exposing their knowledge or identity. Layer: 8 (individual cognition). Related concepts: Reactance, justice sensitivity. Mitigations: Pause-and-verify protocols or ignoring low-stakes errors to avoid traps.",
"CAT-2024-008": "Positive Test Strategy is a cognitive bias where individuals preferentially seek evidence that confirms their hypotheses while neglecting disconfirming information. This vulnerability is rooted in confirmation bias and lazy reasoning, allowing attackers to reinforce false narratives by providing selective supporting data. Mechanism: Hypothesis-testing heuristics favor 'yes' instances over exhaustive falsification (Popperian logic). Exploitation examples: Disinformation campaigns supplying cherry-picked 'proof' for conspiracy theories, leading targets to self-reinforce without seeking counterevidence. Layer: 8 (individual cognition). Related concepts: Confirmation bias, selective exposure. Mitigations: Active open-mindedness training or tools forcing balanced search queries.",
"CAT-2024-010": "Hyperstition is the process by which ideas or beliefs, through collective acceptance and action, become self-fulfilling prophecies in reality. This vulnerability exploits the social construction of reality and memetic spread, turning fiction into fact via behavioral amplification. Mechanism: Feedback loops where belief influences actions, altering outcomes to match the belief (e.g., economic panics). Exploitation examples: Spreading rumors of market crashes to trigger actual selling, or viral prophecies of social unrest leading to riots. Layer: 8 (individual) to 9 (group). Related concepts: Self-fulfilling prophecy, meme theory. Mitigations: Critical media literacy and fact-checking to break the loop early.",
"CAT-2023-003": "Sensitive Information Disclosure is the unintentional release of confidential data due to human error, social engineering, or poor judgment. This vulnerability arises from over-trust, habit, or lack of awareness, allowing attackers to elicit secrets through manipulation. Mechanism: Failure in information security hygiene, often via cognitive overload or reciprocity bias. Exploitation examples: Phishing emails posing as trusted sources to extract credentials, or oversharing in social contexts. Layer: 8 (individual cognition). Related concepts: Social engineering, data leakage. Mitigations: Two-factor authentication and training on need-to-know principles.",
"CAT-2023-001": "Overreliance on Automation is the vulnerability where individuals excessively trust automated systems, leading to complacency and reduced vigilance. This stems from automation bias and deskilling, making targets vulnerable to manipulated AI outputs. Mechanism: Delegation of cognitive effort to machines, reducing human oversight. Exploitation examples: AI-generated deepfakes or biased recommendations influencing decisions without verification. Layer: 8 (individual). Related concepts: Automation bias, complacency. Mitigations: Human-in-the-loop designs and regular manual audits.",
"CAT-2022-321": "Impulsivity is the tendency to act on immediate urges without considering consequences, driven by low self-control or high emotional arousal. This vulnerability allows attackers to exploit time-sensitive bait or emotional triggers. Mechanism: Dysregulation in prefrontal cortex inhibiting delayed gratification. Exploitation examples: Clickbait scams prompting immediate clicks on malicious links. Layer: 8 (individual cognition). Related concepts: Hyperbolic discounting, emotional manipulation. Mitigations: Mindfulness training and delayed response protocols.",
"CAT-2022-293": "Micro Expression is the brief, involuntary facial cue that reveals hidden emotions or intentions, exploitable through observation or AI detection. This vulnerability exposes subconscious states, allowing attackers to gauge reactions. Mechanism: Universal facial muscle movements (Ekman theory) lasting <0.5 seconds. Exploitation examples: Poker bluff detection or AI interviewing to spot lies. Layer: 8 (individual cognition). Related concepts: Body language, nonverbal leakage. Mitigations: Facial control training or masked interactions.",
"CAT-2022-269": "Neoteny is the retention of juvenile traits into adulthood, making individuals more trusting or dependent, exploitable in social engineering. This vulnerability taps into evolutionary cues for care and compliance. Mechanism: Paedomorphic features eliciting protective or submissive responses. Exploitation examples: Using child-like voices or appearances in scams to lower guards. Layer: 8 (individual). Related concepts: Cuteness aggression, paternalism. Mitigations: Awareness of manipulation cues and skepticism toward vulnerability signals.",
"CAT-2022-268": "Need is the basic human drive for resources, safety, or belonging, making individuals vulnerable to offers that promise fulfillment. This vulnerability is exploited through deprivation or false promises. Mechanism: Maslow's hierarchy, where unmet needs prioritize survival over rationality. Exploitation examples: Pyramid schemes targeting financial needs or cults offering belonging. Layer: 8 (individual cognition). Related concepts: Scarcity principle, motivation theory. Mitigations: Self-sufficiency training and need assessment tools.",
"CAT-2022-266": "Fear is the emotional response to perceived threats, amplifying reactivity and impairing judgment. This vulnerability allows attackers to induce panic for compliance or distraction. Mechanism: Amygdala activation overriding prefrontal rational thinking. Exploitation examples: Fear-mongering in disinformation to drive voting or purchasing behavior. Layer: 8 (individual). Related concepts: Terror management theory, fight-or-flight. Mitigations: Exposure therapy and calm reasoning practices.",
"CAT-2022-227": "Unity is the drive for group cohesion and solidarity, exploitable to foster 'us vs. them' dynamics or suppress dissent. This vulnerability leads to blind loyalty. Mechanism: Social identity theory, where in-group bonds override individual judgment. Exploitation examples: Cults or extremist groups using unity rituals to enforce conformity. Layer: 8 (individual). Related concepts: Ingroup bias, conformity. Mitigations: Critical thinking and diverse social networks.",
"CAT-2022-226": "Social Proof is the tendency to conform to what others do or believe, especially in uncertainty. This vulnerability is exploited through fake consensus. Mechanism: Informational social influence, assuming majority is correct. Exploitation examples: Fake reviews or bot-amplified trends to sway opinions. Layer: 8 (individual). Related concepts: Conformity, bandwagon effect. Mitigations: Independent verification and source evaluation.",
"CAT-2022-225": "Scarcity is the perception of limited resources, driving urgency and value inflation. This vulnerability leads to impulsive decisions. Mechanism: Loss aversion amplifying desire for rare items. Exploitation examples: Limited-time offers in scams or flash sales. Layer: 8 (individual). Related concepts: FOMO (fear of missing out), supply-demand manipulation. Mitigations: Time delays and abundance mindset training.",
"CAT-2022-223": "Reciprocity-Need for is the obligation to repay favors, gifts, or concessions. This vulnerability exploits social norms for compliance. Mechanism: Norm of reciprocity creating debt-like feelings. Exploitation examples: Free samples leading to purchases or information sharing. Layer: 8 (individual). Related concepts: Door-in-the-face technique, gift bias. Mitigations: Awareness of manipulation and no-obligation mindset.",
"CAT-2022-220": "Liking is the preference for people or things we find attractive, similar, or complimentary. This vulnerability lowers guards in interactions. Mechanism: Similarity-attraction theory boosting trust. Exploitation examples: Social engineering with flattery or shared interests. Layer: 8 (individual). Related concepts: Halo effect, rapport building. Mitigations: Objective evaluation and separation of personal from professional.",
"CAT-2022-216": "Authority-Deference to is the tendency to obey perceived authority figures without question. This vulnerability enables command exploitation. Mechanism: Milgram obedience dynamics, conditioned from childhood. Exploitation examples: Fake boss emails in BEC scams. Layer: 8 (individual). Related concepts: Obedience bias, power dynamics. Mitigations: Authority verification protocols and critical questioning.",
"CAT-2022-215": "Assistance-Need to Provide is the drive to help others, especially in distress, exploitable through fabricated emergencies. Mechanism: Altruism norms overriding caution. Exploitation examples: Charity scams or bystander manipulation. Layer: 8 (individual). Related concepts: Helper's high, empathy bias. Mitigations: Verify requests and set boundaries.",
"CAT-2022-214": "Network Affect Contagion is the spread of emotions through social networks, amplifying collective moods. This vulnerability enables viral panic or euphoria. Mechanism: Emotional contagion via mimicry and empathy. Exploitation examples: Social media campaigns spreading fear. Layer: 7-8 (network to individual). Related concepts: Mood contagion, viral marketing. Mitigations: Media breaks and emotional regulation.",
"CAT-2022-210": "Sympathy is the emotional sharing of another's distress, leading to vulnerability through manipulated empathy. Mechanism: Mirror neurons activating shared feelings. Exploitation examples: Sob story scams eliciting donations. Layer: 8 (individual). Related concepts: Empathy-altruism hypothesis. Mitigations: Rational empathy balance.",
"CAT-2022-209": "Streisand Effect is the unintended amplification of information through suppression attempts. This vulnerability backfires on censors. Mechanism: Curiosity and reactance increasing interest. Exploitation examples: Leaks gaining traction from takedown notices. Layer: 8 (individual). Related concepts: Forbidden fruit effect, reactance. Mitigations: Strategic silence or positive reframing.",
"CAT-2022-208": "Stereotyping is the application of generalized beliefs to groups, reducing complexity but enabling bias. Mechanism: Cognitive shortcut for categorization. Exploitation examples: Propaganda reinforcing negative stereotypes. Layer: 8 (individual). Related concepts: Prejudice, implicit bias. Mitigations: Contact theory and diversity exposure.",
"CAT-2022-207": "Social Desirability Bias is the tendency to present oneself favorably, distorting truth. Mechanism: Approval-seeking in social contexts. Exploitation examples: Surveys manipulated by leading questions. Layer: 8 (individual). Related concepts: Impression management. Mitigations: Anonymous responses.",
"CAT-2022-206": "Outgroup Homogeneity Bias is perceiving outgroups as more similar than they are. Mechanism: Limited exposure leading to overgeneralization. Exploitation examples: Dehumanizing enemies in war propaganda. Layer: 8 (individual). Related concepts: Ingroup favoritism. Mitigations: Intergroup contact.",
"CAT-2022-205": "Network Manipulated Affect is artificial emotional spread through networks. Mechanism: Algorithmic amplification of affect. Exploitation examples: Bot-driven outrage campaigns. Layer: 8 (individual). Related concepts: Emotional contagion. Mitigations: Algorithm transparency.",
"CAT-2022-204": "Mass Psychogenic Illness is group symptom manifestation without physical cause. Mechanism: Suggestibility and social influence. Exploitation examples: Hoax virus scares. Layer: 8 (individual). Related concepts: Hysteria. Mitigations: Information verification.",
"CAT-2022-203": "Ingroup Bias is favoring one's own group. Mechanism: Social identity theory. Exploitation examples: Tribalism in politics. Layer: 8 (individual). Related concepts: Outgroup derogation. Mitigations: Superordinate goals.",
"CAT-2022-198": "Interoceptive Bias is misinterpretation of bodily signals. Mechanism: Poor body awareness. Exploitation examples: Placebo/nocebo effects in scams. Layer: 8 (individual). Related concepts: Somatic marker hypothesis. Mitigations: Mindfulness.",
"CAT-2022-196": "Prevalence Paradox is misjudging common threats due to familiarity. Mechanism: Availability heuristic. Exploitation examples: Underestimating everyday risks. Layer: 8 (individual). Related concepts: Normalcy bias. Mitigations: Statistical literacy.",
"CAT-2022-192": "Focusing Effect is overweighting salient information. Mechanism: Attention bias. Exploitation examples: Highlighting minor details in misinformation. Layer: 8 (individual). Related concepts: Anchoring. Mitigations: Holistic evaluation.",
"CAT-2022-190": "Boredom is a state of understimulation leading to risk-seeking. Mechanism: Arousal theory. Exploitation examples: Gambling apps targeting bored users. Layer: 8 (individual). Related concepts: Sensation seeking. Mitigations: Productive outlets.",
"CAT-2022-189": "Automaticity is habitual actions without awareness. Mechanism: System 1 thinking. Exploitation examples: Habit-based phishing. Layer: 8 (individual). Related concepts: Heuristics. Mitigations: Habit disruption.",
"CAT-2022-186": "Perceptual Deception is misinterpretation of sensory input. Mechanism: Illusions and biases in perception. Exploitation examples: Optical illusions in scams. Layer: 8 (individual). Related concepts: Gestalt principles. Mitigations: Second looks.",
"CAT-2022-181": "Ignorance is lack of knowledge, exploitable through misinformation. Mechanism: Knowledge gaps filled with assumptions. Exploitation examples: Fake news targeting uninformed. Layer: 8 (individual). Related concepts: Dunning-Kruger. Mitigations: Education.",
"CAT-2022-150": "Complaining-Tendency is habitual negative expression, exploitable for division. Mechanism: Venting reinforcement. Exploitation examples: Forums amplifying dissatisfaction. Layer: 8 (individual). Related concepts: Negativity bias. Mitigations: Positive focus.",
"CAT-2022-143": "Data Vulnerability is exposure of sensitive data. Mechanism: Poor hygiene. Exploitation examples: Data breaches. Layer: 8 (individual). Related concepts: Privacy paradox. Mitigations: Encryption.",
"CAT-2022-121": "Transmission Error is errors in information passing. Mechanism: Communication breakdowns. Exploitation examples: Misinformation spread. Layer: 8 (individual). Related concepts: Telephone game. Mitigations: Verification.",
"CAT-2022-120": "Mis-Addressed Email is sending to wrong recipient. Mechanism: Human error. Exploitation examples: Data leaks. Layer: 8 (individual). Related concepts: Typo squatting. Mitigations: Address confirmation.",
"CAT-2022-119": "Loss Error is data loss. Mechanism: Storage failures. Exploitation examples: Ransomware. Layer: 8 (individual). Related concepts: Backup neglect. Mitigations: Redundancy.",
"CAT-2022-118": "Leakage Errors are unintended data exposure. Mechanism: Oversight. Exploitation examples: Metadata leaks. Layer: 8 (individual). Related concepts: Side-channel attacks. Mitigations: Sanitization.",
"CAT-2022-117": "Disposal Errors are improper data destruction. Mechanism: Incomplete deletion. Exploitation examples: Dumpster diving. Layer: 8 (individual). Related concepts: Data remanence. Mitigations: Shredding.",
"CAT-2022-116": "Configuration Error is misconfigured systems. Mechanism: Human setup mistakes. Exploitation examples: Open ports. Layer: 8 (individual). Related concepts: Default settings bias. Mitigations: Automation.",
"CAT-2022-115": "Curiosity is the drive to explore unknown. Mechanism: Information gap theory. Exploitation examples: Clickbait. Layer: 8 (individual). Related concepts: FOMO. Mitigations: Restraint.",
"CAT-2022-114": "Zeigarnik Effect is better memory for incomplete tasks. Mechanism: Cognitive closure need. Exploitation examples: Cliffhangers in phishing. Layer: 8 (individual). Related concepts: Open loops. Mitigations: Task completion.",
"CAT-2022-113": "Whorfianism is language shaping thought. Mechanism: Linguistic relativity. Exploitation examples: Euphemisms in propaganda. Layer: 8 (individual). Related concepts: Sapir-Whorf hypothesis. Mitigations: Multilingualism.",
"CAT-2022-112": "von Restorff Effect is better recall of distinctive items. Mechanism: Isolation effect. Exploitation examples: Highlighting key disinformation. Layer: 8 (individual). Related concepts: Salience bias. Mitigations: Uniform presentation.",
"CAT-2022-109": "Subjective Validation is accepting vague statements as personal. Mechanism: Barnum effect. Exploitation examples: Horoscopes. Layer: 8 (individual). Related concepts: Forer effect. Mitigations: Specificity testing.",
"CAT-2022-107": "Status Quo Bias is preference for current state. Mechanism: Loss aversion. Exploitation examples: Resistance to change campaigns. Layer: 8 (individual). Related concepts: Inertia. Mitigations: Change framing.",
"CAT-2022-106": "Spotlight Effect is overestimating others' attention. Mechanism: Egocentrism. Exploitation examples: Paranoia induction. Layer: 8 (individual). Related concepts: Imaginary audience. Mitigations: Perspective-taking.",
"CAT-2022-104": "Source Monitoring Error is confusing memory sources. Mechanism: Source amnesia. Exploitation examples: Misattributed quotes in misinformation. Layer: 8 (individual). Related concepts: Cryptomnesia. Mitigations: Source tracking.",
"CAT-2022-103": "Serial Position Effect is better recall of first/last items. Mechanism: Primacy/recency. Exploitation examples: Key messages at start/end. Layer: 8 (individual). Related concepts: Memory curves. Mitigations: Repetition.",
"CAT-2022-102": "Self-Serving Bias is attributing success internally, failure externally. Mechanism: Ego protection. Exploitation examples: Victim-blaming narratives. Layer: 8 (individual). Related concepts: Attribution theory. Mitigations: Reflection.",
"CAT-2022-101": "Self-Relevance Effect is better memory for self-related info. Mechanism: Self-reference. Exploitation examples: Personalized scams. Layer: 8 (individual). Related concepts: Self-schema. Mitigations: Impersonalization.",
"CAT-2022-099": "Satisficing is accepting 'good enough' solutions. Mechanism: Bounded rationality. Exploitation examples: Low-effort scams. Layer: 8 (individual). Related concepts: Heuristics. Mitigations: Thorough search.",
"CAT-2022-098": "Risk Homeostasis is adjusting behavior to maintain risk level. Mechanism: Wilde's theory. Exploitation examples: Safety illusions leading to risk-taking. Layer: 8 (individual). Related concepts: Moral hazard. Mitigations: Risk awareness.",
"CAT-2022-097": "Relativism is judging by relative standards. Mechanism: Comparative thinking. Exploitation examples: Normalizing bad behavior. Layer: 8 (individual). Related concepts: Moral relativism. Mitigations: Absolute standards.",
"CAT-2022-096": "Probability Blindness is poor probability judgment. Mechanism: Innumeracy. Exploitation examples: Lottery scams. Layer: 8 (individual). Related concepts: Gambler's fallacy. Mitigations: Stats training.",
"CAT-2022-094": "Planning Fallacy is underestimating task time. Mechanism: Optimism bias. Exploitation examples: Deadline traps. Layer: 8 (individual). Related concepts: Hofstadter's law. Mitigations: Historical data.",
"CAT-2022-093": "Peak-End Rule is judging experiences by peak/end. Mechanism: Memory bias. Exploitation examples: Ending messages strongly. Layer: 8 (individual). Related concepts: Recency effect. Mitigations: Holistic recall.",
"CAT-2022-092": "Overconfidence is excessive belief in own judgment. Mechanism: Metacognitive illusion. Exploitation examples: Confidence tricks. Layer: 8 (individual). Related concepts: Dunning-Kruger. Mitigations: Calibration.",
"CAT-2022-091": "Optimism Bias is expecting positive outcomes. Mechanism: Motivational bias. Exploitation examples: Risky investments. Layer: 8 (individual). Related concepts: Unrealistic optimism. Mitigations: Pessimism planning.",
"CAT-2022-090": "Omission Bias is preferring harms of inaction. Mechanism: Status quo preference. Exploitation examples: Vaccine hesitancy. Layer: 8 (individual). Related concepts: Default bias. Mitigations: Action framing.",
"CAT-2022-089": "No description available in current search - expand based on CAT wiki if needed.",
"CAT-2022-088": "Neglect of Probability is ignoring base rates. Mechanism: Insensitivity to likelihood. Exploitation examples: Terror hype. Layer: 8 (individual). Related concepts: Base rate fallacy. Mitigations: Bayesian thinking.",
"CAT-2022-087": "Negativity Bias is stronger weight on negative info. Mechanism: Evolutionary survival. Exploitation examples: Fear-mongering. Layer: 8 (individual). Related concepts: Loss aversion. Mitigations: Positive focus.",
"CAT-2022-084": "Mother Teresa Effect is mood improvement from helping. Mechanism: Altruism reward. Exploitation examples: Charity scams. Layer: 8 (individual). Related concepts: Helper's high. Mitigations: Verify causes.",
"CAT-2022-083": "Mood-Congruent Memory is better recall in matching mood. Mechanism: State-dependent learning. Exploitation examples: Negative moods amplifying bad memories. Layer: 8 (individual). Related concepts: Mood bias. Mitigations: Mood management.",
"CAT-2022-082": "Mental Set is fixed approach to problems. Mechanism: Perseveration. Exploitation examples: Locked-in scams. Layer: 8 (individual). Related concepts: Functional fixedness. Mitigations: Lateral thinking.",
"CAT-2022-079": "Loss Aversion is stronger pain from losses. Mechanism: Prospect theory. Exploitation examples: Sunk cost fallacies. Layer: 8 (individual). Related concepts: Endowment effect. Mitigations: Framing gains.",
"CAT-2022-078": "Levels-of-Processing Effect is deeper processing better memory. Mechanism: Craik-Lockhart. Exploitation examples: Semantic tricks. Layer: 8 (individual). Related concepts: Elaboration likelihood. Mitigations: Surface-level caution.",
"CAT-2022-077": "Leveling and Sharpening is memory distortion over time. Mechanism: Serial reproduction. Exploitation examples: Rumor evolution. Layer: 8 (individual). Related concepts: Telephone game. Mitigations: Written records.",
"CAT-2022-073": "Illusory Correlation is perceiving nonexistent relationships. Mechanism: Apophenia. Exploitation examples: Conspiracy theories. Layer: 8 (individual). Related concepts: Pareidolia. Mitigations: Statistical checks.",
"CAT-2022-072": "Illusion of Control is overestimating personal influence. Mechanism: Agency bias. Exploitation examples: Gambling illusions. Layer: 8 (individual). Related concepts: Locus of control. Mitigations: Randomness awareness.",
"CAT-2022-071": "IKEA Effect is valuing self-made items more. Mechanism: Effort justification. Exploitation examples: DIY scams. Layer: 8 (individual). Related concepts: Sunk cost. Mitigations: Objective valuation.",
"CAT-2022-070": "Hyperbolic Discounting is preferring immediate rewards. Mechanism: Temporal discounting. Exploitation examples: Impulse buys. Layer: 8 (individual). Related concepts: Present bias. Mitigations: Commitment devices.",
"CAT-2022-069": "Hindsight Bias is seeing past events as predictable. Mechanism: Knew-it-all-along. Exploitation examples: Post-event manipulation. Layer: 8 (individual). Related concepts: Retrospective bias. Mitigations: Pre-registration.",
"CAT-2022-068": "Halo Effect is overall impression influencing specifics. Mechanism: Cognitive generalization. Exploitation examples: Celebrity endorsements. Layer: 8 (individual). Related concepts: Horn effect. Mitigations: Trait separation.",
"CAT-2022-067": "Gambler's Fallacy is expecting reversal after streak. Mechanism: Independence of events. Exploitation examples: Lottery myths. Layer: 8 (individual). Related concepts: Monte Carlo fallacy. Mitigations: Probability education.",
"CAT-2022-066": "Fundamental Attribution Error is overemphasizing disposition. Mechanism: Attribution theory. Exploitation examples: Character assassinations. Layer: 8 (individual). Related concepts: Actor-observer bias. Mitigations: Situational awareness.",
"CAT-2022-065": "Functional Fixedness is seeing objects only in usual function. Mechanism: Mental set. Exploitation examples: Improvised weapon oversight. Layer: 8 (individual). Related concepts: Einstellung effect. Mitigations: Creative thinking.",
"CAT-2022-064": "Frequency Illusion is noticing something more after awareness. Mechanism: Baader-Meinhof. Exploitation examples: Targeted ads. Layer: 8 (individual). Related concepts: Selection bias. Mitigations: Attention tracking.",
"CAT-2022-063": "Framing Effect is decision influenced by presentation. Mechanism: Prospect theory. Exploitation examples: Positive/negative spin. Layer: 8 (individual). Related concepts: Anchoring. Mitigations: Reframing exercises.",
"CAT-2022-060": "False Uniqueness Bias is overestimating own rarity. Mechanism: Ego-centrism. Exploitation examples: Exclusive scam offers. Layer: 8 (individual). Related concepts: False consensus. Mitigations: Comparative data.",
"CAT-2022-059": "False Memory is recalling nonexistent events. Mechanism: Suggestibility. Exploitation examples: Eyewitness manipulation. Layer: 8 (individual). Related concepts: Mandela effect. Mitigations: Source monitoring.",
"CAT-2022-058": "False Consensus Effect is overestimating agreement. Mechanism: Projection bias. Exploitation examples: Echo chamber amplification. Layer: 8 (individual). Related concepts: Pluralistic ignorance. Mitigations: Diverse polling.",
"CAT-2022-057": "Endowment Effect is valuing owned items more. Mechanism: Ownership bias. Exploitation examples: Auction traps. Layer: 8 (individual). Related concepts: Loss aversion. Mitigations: Willingness-to-pay tests.",
"CAT-2022-056": "Egocentric Bias is self-centered perspective. Mechanism: Anchoring on own view. Exploitation examples: Empathy gaps in scams. Layer: 8 (individual). Related concepts: Curse of knowledge. Mitigations: Perspective-taking.",
"CAT-2022-054": "Dunning–Kruger Effect is incompetents overestimating ability. Mechanism: Metacognitive failure. Exploitation examples: Dunning targets in cons. Layer: 8 (individual). Related concepts: Overconfidence. Mitigations: Expertise feedback.",
"CAT-2022-053": "Dread Aversion is avoiding feared outcomes disproportionately. Mechanism: Anticipatory anxiety. Exploitation examples: Insurance scams. Layer: 8 (individual). Related concepts: Risk aversion. Mitigations: Probability calibration.",
"CAT-2022-052": "Default Bias is preferring pre-selected options. Mechanism: Inertia. Exploitation examples: Opt-out traps. Layer: 8 (individual). Related concepts: Status quo bias. Mitigations: Active choice designs.",
"CAT-2022-049": "Context Dependent Memory is better recall in original context. Mechanism: Encoding specificity. Exploitation examples: Trigger-based recalls in manipulation. Layer: 8 (individual). Related concepts: State-dependent learning. Mitigations: Diverse encoding.",
"CAT-2022-048": "Confirmation Bias is seeking confirming evidence. Mechanism: Motivated reasoning. Exploitation examples: Echo chambers. Layer: 8 (individual). Related concepts: Selective exposure. Mitigations: Disconfirming search.",
"CAT-2022-046": "Cognitive Dissonance is discomfort from conflicting beliefs. Mechanism: Festinger theory. Exploitation examples: Forced compliance. Layer: 8 (individual). Related concepts: Justification. Mitigations: Dissonance resolution.",
"CAT-2022-044": "Clustering Illusion is seeing patterns in random data. Mechanism: Apophenia. Exploitation examples: Conspiracy theories. Layer: 8 (individual). Related concepts: Pareidolia. Mitigations: Statistical testing.",
"CAT-2022-041": "Belief Bias is judging arguments by belief consistency. Mechanism: Prior bias. Exploitation examples: Partisan news. Layer: 8 (individual). Related concepts: Motivated reasoning. Mitigations: Logic training.",
"CAT-2022-040": "Base Rate Neglect is ignoring statistical base rates. Mechanism: Representativeness heuristic. Exploitation examples: Rare event hype. Layer: 8 (individual). Related concepts: Bayes' theorem. Mitigations: Base rate reminders.",
"CAT-2022-034": "Actor-Observer Bias is attributing own actions to situation, others to disposition. Mechanism: Attribution asymmetry. Exploitation examples: Blame shifting. Layer: 8 (individual). Related concepts: Fundamental attribution error. Mitigations: Perspective swap.",
"CAT-2022-027": "Fluency Effect is easier processing feels truer. Mechanism: Processing fluency. Exploitation examples: Repetition in lies. Layer: 8 (individual). Related concepts: Illusion of truth. Mitigations: Effortful thinking.",
"CAT-2022-026": "Familiarity is preference for known things. Mechanism: Mere exposure effect. Exploitation examples: Brand loyalty scams. Layer: 8 (individual). Related concepts: Recognition heuristic. Mitigations: Novelty exposure.",
"CAT-2022-236": "Legal Loophole is exploitable gaps in laws. Mechanism: Regulatory arbitrage. Exploitation examples: Tax evasion schemes. Layer: 10 (systemic). Related concepts: Letter vs. spirit. Mitigations: Law patching.",
"CAT-2022-233": "Shadow Security is unofficial practices bypassing controls. Mechanism: Workaround culture. Exploitation examples: Insider threats. Layer: 9 (group). Related concepts: Shadow IT. Mitigations: Policy enforcement.",
"CAT-2022-232": "Shadow IT is unauthorized technology use. Mechanism: Convenience over compliance. Exploitation examples: Unsecured tools. Layer: 9 (group). Related concepts: BYOD risks. Mitigations: Approved alternatives.",
"CAT-2022-231": "Shadow AP is rogue access points. Mechanism: Unauthorized networks. Exploitation examples: Man-in-the-middle. Layer: 9 (group). Related concepts: WiFi phishing. Mitigations: Network scanning."
}

# Precompute embeddings at module load
print("Precomputing TTP embeddings with multi-qa-mpnet-base-dot-v1...")
texts = list(TTP_DESCRIPTIONS.values())
embeddings = embed_texts(texts)
TTP_EMBEDDINGS = dict(zip(TTP_DESCRIPTIONS.keys(), embeddings))
print("TTP embeddings ready.")


def cluster_ttp(ttp_ids: List[str]) -> Dict[str, any]:
    """
    Given a list of TTP IDs from LLM extraction or CVF, compute:
    - centroid embedding
    - nearest known TTP cluster (by cosine similarity)
    - similarity scores to all known TTPs
    """

    if not ttp_ids:
        return {
            "cluster_id": None,
            "similarities": {},
            "centroid": None,
            "activated_count": 0
        }

    vectors = []
    valid_ids = []
    for ttp in ttp_ids:
        if ttp in TTP_EMBEDDINGS:
            vectors.append(TTP_EMBEDDINGS[ttp])
            valid_ids.append(ttp)
        else:
            print(f"Warning: Unknown TTP '{ttp}' — skipping in clustering")

    if not vectors:
        return {
            "cluster_id": None,
            "similarities": {},
            "centroid": None,
            "activated_count": 0
        }

    vectors = np.array(vectors)
    centroid = np.mean(vectors, axis=0)

    # Cosine similarity (safe with normalized embeddings)
    similarities = {}
    for key, emb in TTP_EMBEDDINGS.items():
        sim = float(np.dot(centroid, emb) / (np.linalg.norm(centroid) * np.linalg.norm(emb)))
        similarities[key] = round(sim, 3)

    cluster_id = max(similarities, key=similarities.get)
    top_similarity = similarities[cluster_id]

    return {
        "cluster_id": cluster_id,
        "top_similarity": top_similarity,
        "similarities": similarities,
        "centroid": centroid.tolist(),
        "activated_count": len(valid_ids),
        "activated_ttps": valid_ids
    }