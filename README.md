# NAM-D Diagnostic Engine

### The NAM-D Diagnostic Engine is a proprietary AI-powered program for dissecting and analyzing the risk posed by a given propaganda narrative. NAM-D, or Narrative Access and Maneuver Denial is a strategy I built doing research on Information Warfare and its components. 

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
Run in CLI with `python ./execute` when all dependencies and packages are installed. The program will automatically prime the TTP engine with the necessary TTPs. Enter your desired narrative when prompted. Target Audience demographics currently do nothing so just enter a string (or integer for age range) and let the model do it's work. You should see a raw JSON output summarzing the narrative followed by a PMESII Radar Graph and TTP Radar Graph. The CLI will then also have a detailed report of the findings. Enjoy!

# Next Steps
Next up I'm integrating TA Analysis as a dynamic value to factor into Risk Analysis and hooking the whole thing up to a front-end FUI. That will finish up NDE V 1.0. Stay Posted for updates!
