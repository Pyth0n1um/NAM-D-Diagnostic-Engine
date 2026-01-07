# NAM-D Diagnostic Engine

### The NAM-D Diagnostic Engine is a proprietary AI-powered program for dissecting and analyzing the risk posed by a given propaganda narrative. NAM-D, or Narrative Access and Maneuver Denial, is a strategy I developed while researching Information Warfare and its components. 

# Dependencies
- plotly
- matploylib
- huggingface-hub
- sentencetransformers
- llama.cpp
- openai
- Provided nar_new.gbnf grammar file (for formatting llama server outputs)
- allura-forge_Llama-3.3-8B-Instruct-Q4_K_M.gguf (saved to llama.cpp/models)

# Instructions
Run in CLI with `python ./execute` when all dependencies and packages are installed. The program will automatically prime the TTP engine with the necessary TTPs. Enter your desired narrative when prompted. Target Audience demographics currently do nothing, so just enter a string (or integer for age range) and let the model do its work. You should see a raw JSON output summarizing the narrative, followed by a PMESII Radar Graph, TTP Radar Graph, and TTP Radar Graph. The CLI will then also have a detailed report of the findings. Enjoy!

# Next Steps
I'm currently working on an updated TA Analysis to accept values from the front end. For the time being, NDE v1.0 is done, but v1.1 is on the way, so stay tuned.
