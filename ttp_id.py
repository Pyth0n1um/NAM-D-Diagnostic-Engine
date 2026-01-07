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
# TTP Registry (Structured with id, name, description)
# -----------------------------

TTP_DESCRIPTIONS = {
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

# TTP Registry (new)
ttp_texts = [entry["description"] for entry in TTP_DESCRIPTIONS.values()]
ttp_embeddings = embed_texts(ttp_texts)
TTP_EMBEDDINGS = {entry["id"]: emb for entry, emb in zip(TTP_DESCRIPTIONS.values(), ttp_embeddings)}

print(f"Embeddings ready:")
print(f"  - TTP registry: {len(TTP_EMBEDDINGS)} TTPs")

def identify_ttp_dual(
    narrative_text: str,
    top_k_per_registry: int = 5,
    min_confidence: float = 0.45,
    prioritize_cognitive: bool = True
) -> List[Dict[str, Any]]:
    if not narrative_text.strip():
        return []

    narrative_emb = embed_texts([narrative_text])[0]
    narrative_lower = narrative_text.lower()

    # --- Propaganda Style Indicators ---
    propaganda_indicators = [
        "hero", "victim", "enemy", "threat", "crisis", "urgent", "now", "save", "protect",
        "regime", "elite", "corrupt", "people", "nation", "pride", "betrayal"
    ]

    # --- Penalty Keywords ---
    penalty_keywords = [
        "ultrasonic", "model inversion", "backdoor", "prompt injection",
        "coercion", "torture", "extract keys", "rubber-hose", "cryptanalysis",
        "encryption", "blockchain", "quantum", "cryptography", "malware", "trojan",
        "wireless", "bluetooth", "wi-fi", "5g", "satellite", "shipping", "kidnap", "kidnapping",
        "hostage", "smuggle", "smuggling", "drone", "uav", "phishing", "ransomware", "spyware",
        "zero-day", "email"
    ]
    general_disinfo_keywords = [
        "manipulate", "deceive", "amplify", "distort", "mislead", "false", "fake", "inauthentic",
        "control", "influence", "propaganda", "narrative", "frame", "spin"
    ]

    # --- Targeted Proword Rewards ---
    prowords_deception = [
        "multiple", "layered", "simultaneous", "coordinated", "overlapping", "campaigns",
        "vectors", "channels", "various", "reports"
    ]
    prowords_narrative = [
        "story", "tale", "legend", "myth", "frame", "shape belief", "heroic", "victim",
        "journey", "destiny", "narrative", "reports", "footage"
    ]

    # --- TTP Registry Matching ---
    ttp_matches = []
    for ttp_id, ttp_emb in TTP_EMBEDDINGS.items():
        sim = float(np.dot(narrative_emb, ttp_emb))
        entry = TTP_DESCRIPTIONS[ttp_id]

        boost = 0.0

        # Propaganda style boost (slightly lower than CAT)
        if any(ind in narrative_lower for ind in propaganda_indicators):
            boost += 0.1

        # Lighter technical penalty
        tech_keywords_disarm = ["deepfake", "bot", "hashtag", "inauthentic"]
        if any(kw in entry["description"].lower() for kw in tech_keywords_disarm):
            if not any(kw in narrative_lower for kw in tech_keywords_disarm):
                sim *= 0.7

        final_conf = min(1.0, sim + boost)

        if final_conf >= min_confidence:
            ttp_matches.append({
                "id": ttp_id,
                "name": entry["name"],
                "confidence": round(final_conf, 3),
                "source": "DISARM"
            })

    ttp_matches.sort(key=lambda x: x["confidence"], reverse=True)

    # --- Balanced Merge: Interleave Top from Each (No Final Sort) ---
    final_results = []

    for i in range(top_k_per_registry):
        if i < len(ttp_matches):
            final_results.append(ttp_matches[i])

    return final_results
