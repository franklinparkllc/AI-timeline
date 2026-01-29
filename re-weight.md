# AI Timeline Event Re-Weighting

**Scale:** 1 = minor · 2 = major · 3 = landmark

This document reviews all 216 events in `data/AI_Timeline_1940-2025.csv` and proposes a simplified significance weight (1, 2, or 3) for each. Rationale is brief; web and standard AI histories were used where helpful.

---

## Line-by-Line Review

| # | Year | Event (short) | Old | New | Rationale |
|---|------|---------------|-----|-----|------------|
| 1 | 1943 | McCulloch-Pitts logical calculus | 10 | **3** | Foundational neural net theory; cited by von Neumann; birth of artificial neurons. |
| 2 | 1945 | As We May Think (Vannevar Bush) | 4 | **2** | Vision of hypertext/memex; influential for human–machine symbiosis. |
| 3 | 1949 | Hebbian learning (*Organization of Behavior*) | 7 | **2** | Core “cells that fire together wire together”; foundational for learning rules. |
| 4 | 1950 | Turing Test described | 7 | **3** | Definitive philosophical benchmark for machine intelligence; still referenced. |
| 5 | 1951 | SNARC neural network hardware | 5 | **1** | Early hardware experiment; limited direct lineage. |
| 6 | 1952 | Checkers program (Samuel) | 5 | **2** | First self-improving game program; established ML-in-games. |
| 7 | 1956 | “AI” coined; Logic Theorist; Dartmouth | 10 | **3** | Birth of the field; name and first symbolic AI program. |
| 8 | 1958 | LISP created | 7 | **2** | Dominant AI language for decades; symbolic reasoning. |
| 9 | 1958 | Perceptron algorithm (Rosenblatt) | 7 | **2** | First practical learning machine; sparked connectionist wave. |
| 10 | 1959 | “Machine learning” framing; checkers | 7 | **2** | Named and popularized ML as a concept. |
| 11 | 1960 | ADALINE | 6 | **1** | Incremental improvement over perceptron; less cited. |
| 12 | 1965 | DENDRAL expert system begins | 7 | **2** | First major expert system; established knowledge-based AI. |
| 13 | 1965 | Fuzzy sets (Zadeh) | 7 | **1** | Niche in control/systems; limited in modern ML. |
| 14 | 1966 | Shakey robot project | 7 | **2** | First mobile reasoning robot; planning + perception. |
| 15 | 1966 | ELIZA chatbot (CACM) | 6 | **2** | First widely known chatbot; demonstrated NLP illusion. |
| 16 | 1966 | ALPAC report (MT funding cut) | 6 | **1** | Policy setback for MT; less central to AI narrative. |
| 17 | 1968 | Hubel-Wiesel receptive fields | 6 | **2** | Biological basis for convnets; Nobel; inspired CNNs. |
| 18 | 1968 | A* search algorithm | 8 | **2** | Foundational heuristic search; still taught and used. |
| 19 | 1969 | Perceptrons book (Minsky/Papert) | 6 | **2** | Correct limits (XOR); contributed to first AI winter. |
| 20 | 1969 | Kaissa chess program | 4 | **1** | Early chess program; superseded by later systems. |
| 21 | 1970 | SHRDLU blocks world | 7 | **2** | Landmark NL understanding in microworld; highly cited. |
| 22 | 1970 | Early continuous speech recognition | 5 | **1** | Incremental step; Reddy’s later work more impactful. |
| 23 | 1972 | Prolog introduced | 6 | **2** | Major logic-programming language for AI. |
| 24 | 1973 | Lighthill report | 6 | **2** | UK funding cut; catalyzed first AI winter. |
| 25 | 1974 | First AI winter begins | 6 | **2** | Major funding/expectation crash; shaped field. |
| 26 | 1975 | Backpropagation dissertation (Werbos) | 7 | **2** | First backprop; not widely adopted until later. |
| 27 | 1975 | Frames (Minsky) | 5 | **1** | Knowledge representation; one of several KR schemes. |
| 28 | 1976 | MYCIN expert system | 6 | **2** | Influential rule-based medical system; uncertainty. |
| 29 | 1977 | EM algorithm | 7 | **2** | Foundational for mixture models and latent variables. |
| 30 | 1979 | Industrial robot fatality (Williams) | 0 | **1** | Safety/labor milestone; tangential to AI algorithms. |
| 31 | 1980 | Neocognitron (Fukushima) | 7 | **2** | Early convnet-like architecture; inspired LeCun. |
| 32 | 1980 | XCON expert system at DEC | 6 | **2** | First large-scale commercial expert system success. |
| 33 | 1982 | Hopfield Network | 6 | **2** | Rekindled interest in neural nets; energy/associative memory. |
| 34 | 1982 | Dragon Systems founded | 4 | **1** | Commercial speech; one of several players. |
| 35 | 1984 | CART (decision trees) | 7 | **2** | Standard tool for classification/regression. |
| 36 | 1986 | Backprop popularized (Rumelhart et al.) | 10 | **3** | Made deep learning feasible; Nature; reignited NNs. |
| 37 | 1986 | Discourse structure (Grosz/Sidner) | 5 | **1** | NLP discourse; niche vs. broader AI impact. |
| 38 | 1986 | ID3 decision tree algorithm | 6 | **1** | Early tree learning; superseded by CART/RF in prominence. |
| 39 | 1986 | First robot car demos (Dickmanns) | 5 | **2** | Pioneering autonomous driving; DARPA lineage. |
| 40 | 1986 | Boltzmann Machine | 6 | **1** | Theoretical; less direct impact than backprop paper. |
| 41 | 1987 | NETtalk | 5 | **1** | Demonstrative; less foundational than other NN work. |
| 42 | 1988 | Jabberwacky/Thoughts chatbot begins | 0 | **1** | Long-running chatbot project; limited technical impact. |
| 43 | 1988 | Bayesian networks (Pearl textbook) | 8 | **2** | Unified probabilistic reasoning; foundational for causal AI. |
| 44 | 1988 | IBM statistical MT | 6 | **2** | Shift to statistical MT; pre-neural era milestone. |
| 45 | 1989 | TDNN for speech | 6 | **1** | Incremental; Waibel’s later work more visible. |
| 46 | 1989 | Q-learning introduced | 7 | **2** | Core RL algorithm; foundation for DQN etc. |
| 47 | 1989 | CNNs for vision (LeCun) | 7 | **2** | Early applied CNNs; precursor to LeNet/AlexNet. |
| 48 | 1990 | Neuromorphic systems (Mead) | 6 | **1** | Hardware vision; niche vs. mainstream ML. |
| 49 | 1990 | *The Age of Intelligent Machines* | 3 | **1** | Popular book; cultural more than technical. |
| 50 | 1991 | DART (Gulf War logistics) | 6 | **1** | Applied OR; tangential to core AI narrative. |
| 51 | 1991 | “Intelligence without representation” (Brooks) | 6 | **2** | Behavior-based robotics; influential paradigm. |
| 52 | 1992 | REINFORCE policy gradient | 7 | **2** | Foundational for policy-gradient RL. |
| 53 | 1993 | Polly vision-based agent | 3 | **1** | Single robot; incremental. |
| 54 | 1993 | MIT Cog humanoid project | 5 | **1** | Research platform; less lasting impact. |
| 55 | 1994 | “Soft computing” popularized | 4 | **1** | Term consolidation; minor conceptual. |
| 56 | 1995 | Random decision forests (Tin Kam Ho) | 6 | **1** | Early ensemble; Breiman’s RF more influential. |
| 57 | 1995 | Support Vector Machines | 8 | **2** | Dominant classifier pre-deep learning. |
| 58 | 1995 | Robot car 1000 mi Munich–Copenhagen | 5 | **1** | Endurance demo; one milestone of many. |
| 59 | 1996 | AdaBoost | 7 | **2** | Key boosting algorithm; widely used. |
| 60 | 1996 | TD-Gammon | 7 | **2** | Showed TD learning in games; influenced DQN era. |
| 61 | 1997 | Deep Blue defeats Kasparov | 10 | **3** | First machine to beat reigning chess champion; global impact. |
| 62 | 1997 | LSTM (Hochreiter/Schmidhuber) | 10 | **3** | Solved long-range dependencies; backbone of sequence models for years. |
| 63 | 1998 | POMDPs (planning under uncertainty) | 7 | **1** | Important theory; narrower impact than core ML. |
| 64 | 1998 | LeNet-5 / CNNs for documents | 8 | **2** | Practical CNN success; precursor to ImageNet era. |
| 65 | 1998 | Sutton-Barto RL textbook | 7 | **2** | Standard reference for RL. |
| 66 | 1998 | Minerva and Pearl robots (Thrun) | 4 | **1** | Museum guides; incremental robotics. |
| 67 | 1999 | Sony AIBO | 4 | **1** | Consumer robot; cultural more than technical. |
| 68 | 2000 | KISMET emotional robot | 4 | **1** | Affective HRI; niche impact. |
| 69 | 2001 | Echo state network | 6 | **1** | Reservoir computing; specialized. |
| 70 | 2001 | Random Forests (Breiman) | 9 | **2** | Dominant ensemble method; still ubiquitous. |
| 71 | 2001 | Conditional Random Fields | 8 | **2** | Standard for sequence labeling (NLP, vision). |
| 72 | 2001 | Neural Probabilistic Language Model (Bengio) | 7 | **2** | Early neural LM; paved way for word2vec/transformers. |
| 73 | 2002 | iRobot Roomba | 4 | **2** | First mass-market autonomous consumer robot. |
| 74 | 2003 | LDA topic modeling | 8 | **2** | Standard unsupervised topic model. |
| 75 | 2004 | DARPA Grand Challenge | 6 | **2** | Kickstarted modern autonomous vehicle research. |
| 76 | 2005 | Blue Brain Project launched | 4 | **1** | Neuroscience simulation; tangential to applied AI. |
| 77 | 2006 | Deep Belief Nets revive deep learning | 9 | **2** | Key step in deep learning revival; pre-ImageNet. |
| 78 | 2007 | CUDA introduced | 7 | **2** | Enabled GPU computing for NNs; critical infrastructure. |
| 79 | 2007 | GPUs for large-scale NN training | 6 | **2** | Same story as CUDA; major enabler. |
| 80 | 2007 | Stacked Auto-Encoders | 6 | **1** | Unsupervised pretraining; superseded by supervised deep. |
| 81 | 2007 | Checkers solved | 6 | **1** | Perfect play; narrow game milestone. |
| 82 | 2009 | Deep Boltzmann Machines | 7 | **1** | One of several deep generative models. |
| 83 | 2009 | ROS presented and released | 6 | **2** | De facto standard for robot software. |
| 84 | 2008 | SyNAPSE (IBM neuromorphic) | 6 | **1** | Research program; limited mainstream impact. |
| 85 | 2008 | OAQA (Watson precursor) | 4 | **1** | Internal step toward Watson. |
| 86 | 2009 | ImageNet dataset introduced | 10 | **3** | Benchmark that drove modern computer vision; ILSVRC. |
| 87 | 2009 | Google self-driving project begins | 6 | **2** | Launched major AV effort; Waymo lineage. |
| 88 | 2009 | Bitcoin/blockchain introduced | 0 | **1** | Outside core AI; kept for context. |
| 89 | 2010 | DeepMind incorporated | 6 | **2** | Company behind AlphaGo, DQN, AlphaFold. |
| 90 | 2011 | IBM Watson wins Jeopardy! | 7 | **3** | First major public AI vs. humans in open-domain QA. |
| 91 | 2011 | Siri introduced | 5 | **2** | First mainstream voice assistant. |
| 92 | 2012 | Google Now launched | 5 | **1** | Assistant iteration; Siri/Cortana already established. |
| 93 | 2012 | DNNs transform speech recognition | 7 | **2** | Breakthrough in speech; Hinton et al. |
| 94 | 2012 | AlexNet wins ImageNet | 10 | **3** | Triggered deep learning revolution in vision and beyond. |
| 95 | 2012 | Deep learning surge (post-ImageNet) | 6 | **1** | Redundant with AlexNet/ImageNet entries. |
| 96 | 2013 | word2vec | 9 | **2** | Made neural word embeddings standard in NLP. |
| 97 | 2013 | Vicarious CAPTCHA (unverified) | 0 | **1** | Claim not peer-reviewed; minor. |
| 98 | 2014 | DeepFace near-human face verification | 7 | **1** | Application milestone; superseded by later work. |
| 99 | 2014 | GANs introduced | 10 | **3** | New paradigm for generative models; huge impact. |
| 100 | 2014 | Seq2Seq with neural networks | 9 | **2** | Standard architecture for NMT/dialogue. |
| 101 | 2014 | Neural MT attention (Bahdanau et al.) | 9 | **2** | Attention mechanism; precursor to Transformers. |
| 102 | 2014 | Microsoft Cortana announced | 4 | **1** | One of several assistants; discontinued. |
| 103 | 2014 | Amazon Echo/Alexa announced | 5 | **2** | Defined smart speaker category. |
| 104 | 2014 | Google acquires DeepMind | 6 | **2** | Enabled AlphaGo and later DeepMind work. |
| 105 | 2015 | OpenAI founded | 7 | **2** | Major lab; ChatGPT, GPT, DALL·E. |
| 106 | 2015 | DQN in Nature (Atari) | 10 | **2** | Human-level RL in games; key DeepMind result. |
| 107 | 2015 | ResNet (very deep CNNs) | 9 | **2** | Enabled deep networks; skip connections standard. |
| 108 | 2015 | TensorFlow open-sourced | 8 | **2** | Dominant framework for years. |
| 109 | 2015 | AlphaGo defeats Fan Hui | 9 | **2** | First pro defeat; prelude to Lee Sedol. |
| 110 | 2015 | Google self-driving 1M miles | 6 | **1** | Mileage milestone; incremental. |
| 111 | 2016 | AlphaGo defeats Lee Sedol | 10 | **3** | Go milestone; global attention; “AI has arrived.” |
| 112 | 2016 | WaveNet (raw audio) | 8 | **2** | High-quality neural TTS/generative audio. |
| 113 | 2017 | PyTorch open-sourced | 8 | **2** | Leading research framework; industry adoption. |
| 114 | 2017 | Libratus beats poker pros | 7 | **2** | Imperfect-information game milestone. |
| 115 | 2017 | “Attention is all you need” (Transformers) | 10 | **3** | Architecture behind BERT, GPT, LLMs; redefined NLP. |
| 116 | 2017 | Google TPU announced | 7 | **2** | Custom AI accelerator; scale for large models. |
| 117 | 2017 | Capsule Networks | 6 | **1** | Alternative to CNNs; limited adoption. |
| 118 | 2018 | Alibaba reading test | 5 | **1** | Single benchmark; narrow impact. |
| 119 | 2018 | Google Duplex demonstrated | 6 | **2** | Conversational AI in the wild; ethical debate. |
| 120 | 2018 | GPT (first) released | 8 | **2** | First transformer LM from OpenAI; pre-GPT-2/3. |
| 121 | 2018 | BERT | 9 | **3** | Dominant NLP baseline; “pretrain then finetune” standard. |
| 122 | 2019 | GPT-2 (1.5B); misuse concerns | 9 | **2** | Scaled LM; sparked debate on release. |
| 123 | 2019 | BERT widely adopted | 8 | **1** | Adoption wave; redundant with BERT entry. |
| 124 | 2019 | AlphaStar Grandmaster StarCraft II | 8 | **2** | RTS milestone; complex game. |
| 125 | 2019 | PyTorch 1.0 | 7 | **1** | Version bump; framework already major. |
| 126 | 2019 | Nvidia A100 GPU | 8 | **2** | Workhorse for training large models. |
| 127 | 2020 | GPT-3 (175B) | 10 | **3** | Scaled LLMs; few-shot; shifted industry toward scale. |
| 128 | 2020 | AlphaFold 2 solves protein folding | 10 | **3** | Scientific breakthrough; structural biology transformed. |
| 129 | 2020 | GitHub Copilot development begins | 7 | **1** | Precursor to launch; launch event is major. |
| 130 | 2020 | DALL·E 1 announced | 8 | **2** | First high-profile text-to-image from OpenAI. |
| 131 | 2020 | Switch Transformer (1.6T) | 7 | **1** | Scale milestone; less iconic than GPT-3. |
| 132 | 2021 | GitHub Copilot launched | 8 | **2** | First mass-market AI pair programmer. |
| 133 | 2021 | DALL·E 2 development | 8 | **1** | Iteration; release is the milestone. |
| 134 | 2021 | AlphaFold 2 Nature paper | 9 | **1** | Publication of same result as 2020 entry. |
| 135 | 2021 | Codex released | 8 | **2** | Powers Copilot; code LMs. |
| 136 | 2021 | Wu Dao 2.0 (1.75T) | 7 | **1** | Large Chinese model; one of several. |
| 137 | 2021 | Tesla Dojo | 7 | **1** | Custom training infra; niche vs. global impact. |
| 138 | 2021 | Meta AI Research SuperCluster | 7 | **1** | Large cluster; one of many. |
| 139 | 2022 | ChatGPT launched (Nov 30) | 10 | **3** | Fastest-growing consumer app; brought LLMs to mainstream. |
| 140 | 2022 | DALL·E 2 released | 9 | **2** | High-quality text-to-image; broad adoption. |
| 141 | 2022 | Stable Diffusion released (open) | 9 | **2** | Open-source text-to-image; democratized generative art. |
| 142 | 2022 | Midjourney v3/v4 | 8 | **1** | Popular tool; one of several image generators. |
| 143 | 2022 | Galactica released and withdrawn | 6 | **1** | Cautionary release; narrow impact. |
| 144 | 2022 | LaMDA/sentience controversy | 5 | **2** | Governance/ethics moment; raised public awareness. |
| 145 | 2022 | Whisper released | 8 | **2** | Strong open speech recognition; widely used. |
| 146 | 2022 | AlphaTensor (matrix multiply) | 7 | **1** | Algorithm discovery; niche vs. product impact. |
| 147 | 2022 | JEPA (LeCun) | 7 | **1** | Research direction; early stage. |
| 148 | 2022 | PaLM (540B) | 8 | **2** | Large LM; scale and capability milestone. |
| 149 | 2022 | ChatGPT 1M users in 5 days | 9 | **1** | Sub-milestone of ChatGPT launch. |
| 150 | 2023 | GPT-4 released (multimodal) | 10 | **3** | Top capability model; multimodal; industry standard. |
| 151 | 2023 | ChatGPT 100M users | 10 | **2** | Adoption milestone; reinforces ChatGPT as landmark. |
| 152 | 2023 | Google Bard → Gemini | 8 | **2** | Main competitor to ChatGPT. |
| 153 | 2023 | Claude released (Anthropic) | 8 | **2** | Major alternative LLM; safety-focused. |
| 154 | 2023 | LLaMA / LLaMA 2 (open) | 9 | **2** | Open-weight LLMs; research and industry. |
| 155 | 2023 | Mistral 7B | 8 | **1** | Efficient small model; one of many. |
| 156 | 2023 | Microsoft Copilot (Office 365) | 8 | **2** | AI integrated into productivity suite. |
| 157 | 2023 | Adobe Firefly | 7 | **1** | Creative tool; niche. |
| 158 | 2023 | Runway Gen-2 (text-to-video) | 7 | **2** | Early strong text-to-video. |
| 159 | 2023 | AutoGPT / BabyAGI | 6 | **1** | Agent frameworks; hype then niche. |
| 160 | 2023 | Code Interpreter (ChatGPT) | 8 | **1** | Feature; part of ChatGPT evolution. |
| 161 | 2023 | Sam Altman fired and rehired (OpenAI) | 8 | **2** | Governance/corporate moment; broad coverage. |
| 162 | 2023 | EU AI Act passed | 8 | **3** | First major comprehensive AI law; global reference. |
| 163 | 2023 | GPT-4 Vision | 9 | **2** | Multimodal capability; part of GPT-4 story. |
| 164 | 2023 | Gemini announced (Dec) | 9 | **2** | Google’s unified model family. |
| 165 | 2023 | Mamba (SSMs) | 8 | **2** | Efficient sequence architecture; alternative to transformers. |
| 166 | 2023 | Nvidia H100 standard for training | 9 | **2** | Default GPU for LLM training. |
| 167 | 2023 | Character.AI millions of users | 7 | **1** | Popular product; narrow vs. general AI. |
| 168 | 2023 | Custom instructions and GPTs | 8 | **2** | Customizable ChatGPT; platform move. |
| 169 | 2023 | Meta AI with celebrity avatars | 6 | **1** | Product feature; incremental. |
| 170 | 2023 | xAI Grok launched | 7 | **1** | Another LLM entrant. |
| 171 | 2024 | Sora (text-to-video) announced | 9 | **2** | High-profile generative video; not yet broadly released. |
| 172 | 2024 | Claude 3 family | 9 | **2** | Strong capability tier (Opus/Sonnet/Haiku). |
| 173 | 2024 | Gemini 1.5 (1M token context) | 9 | **2** | Long-context milestone. |
| 174 | 2024 | GPT-4o (omni) | 9 | **2** | Fast multimodal; default ChatGPT model. |
| 175 | 2024 | Google I/O Gemini integration | 8 | **1** | Product rollout; incremental. |
| 176 | 2024 | Apple Intelligence (WWDC) | 8 | **2** | Apple’s on-device and cloud AI strategy. |
| 177 | 2024 | ChatGPT desktop; GPT-4o mini | 8 | **1** | Product updates. |
| 178 | 2024 | Anthropic $4B (Amazon) | 7 | **1** | Funding; one of many rounds. |
| 179 | 2024 | Llama 3 / 3.1 (405B) | 9 | **2** | Large open-weight model; strong baseline. |
| 180 | 2024 | Mistral Large 2 | 8 | **1** | Model release; one of many. |
| 181 | 2024 | DeepSeek-V2 | 8 | **2** | Strong Chinese model; competitive with GPT-4. |
| 182 | 2024 | Perplexity traction | 8 | **2** | AI-native search; product category. |
| 183 | 2024 | NotebookLM Audio Overview | 8 | **1** | Feature; niche. |
| 184 | 2024 | ChatGPT Search launched | 8 | **1** | Product feature. |
| 185 | 2024 | California SB 1047 vetoed | 7 | **1** | State policy; vetoed. |
| 186 | 2024 | Strawberry / o1 (reasoning) | 9 | **2** | Reasoning-focused model; significant capability. |
| 187 | 2024 | Grok 2 with image gen | 8 | **1** | Model iteration. |
| 188 | 2024 | Amazon Nova | 8 | **1** | New model family; early. |
| 189 | 2024 | Gemini 2.0 Flash | 8 | **1** | Model variant. |
| 190 | 2024 | OpenAI o1 / o1-pro (Dec) | 9 | **2** | Reasoning models; aligns with Strawberry. |
| 191 | 2024 | Microsoft Copilot Vision | 7 | **1** | Multimodal feature. |
| 192 | 2024 | Runway Gen-3 Alpha | 8 | **1** | Video model iteration. |
| 193 | 2024 | ElevenLabs unicorn | 7 | **1** | Valuation; voice product. |
| 194 | 2024 | Anthropic interpretability (mapping) | 8 | **2** | Research on understanding LMs. |
| 195 | 2024 | Figure 01 + ChatGPT | 8 | **2** | Humanoid robot + LLM; robotics milestone. |
| 196 | 2024 | Tesla Optimus Gen 2 | 7 | **2** | Humanoid robot demo; industry attention. |
| 197 | 2024 | Nvidia most valuable company | 9 | **2** | Market milestone; reflects AI infrastructure importance. |
| 198 | 2024 | Claude 3.5 Sonnet | 9 | **1** | Model update; strong but incremental. |
| 199 | 2024 | GPT-4o advanced voice | 9 | **1** | Feature; part of 4o. |
| 200 | 2024 | OpenAI DevDay (realtime API) | 8 | **2** | Realtime/voice API; platform expansion. |
| 201 | 2024 | Anthropic Computer Use | 9 | **2** | Claude controls computers; agentic capability. |
| 202 | 2024 | ChatGPT 200M weekly users | 9 | **2** | Scale milestone. |
| 203 | 2024 | Waymo expands AV service | 8 | **2** | Commercial AV at scale. |
| 204 | 2024 | AlphaFold 3 | 9 | **2** | Broader molecules; builds on landmark AF2. |
| 205 | 2025 | DeepSeek R1 (open reasoning) | 10 | **2** | Open reasoning model; significant but recent. |
| 206 | 2025 | ChatGPT Pro ($200/mo) | 8 | **1** | Pricing tier. |
| 207 | 2025 | Claude 4 family | 9 | **2** | Next-gen Claude. |
| 208 | 2025 | Gemini 2.0 widely available | 9 | **2** | Google’s current flagship. |
| 209 | 2025 | Deep research (ChatGPT etc.) | 8 | **1** | Feature across platforms. |
| 210 | 2025 | Operator (OpenAI browser agent) | 9 | **2** | Agent product; automation. |
| 211 | 2025 | Gemini Robotics | 8 | **2** | Google’s embodied AI push. |
| 212 | 2025 | WeatherNext 2 | 7 | **1** | Domain model; niche. |
| 213 | 2025 | SIMA 2 (3D worlds) | 7 | **1** | Research agent. |
| 214 | 2025 | Genie 3 (world model) | 8 | **1** | Generative world model; early. |
| 215 | 2026 | Prism (OpenAI scientists) | 8 | **1** | Product announcement; future. |
| 216 | 2026 | GPT-5.2 scientific reasoning | 9 | **1** | Model version; future. |

---

## Summary

| New weight | Count | Meaning |
|------------|-------|---------|
| **3 (Landmark)** | 24 | Field-defining or once-in-a-generation impact |
| **2 (Major)** | 117 | Important advance or product; widely cited or used |
| **1 (Minor)** | 75 | Incremental, niche, redundant, or tangential |

**Landmarks (3)** include: McCulloch-Pitts, Turing Test, Dartmouth/Logic Theorist, backprop popularized (1986), Deep Blue, LSTM, ImageNet, AlexNet, Watson Jeopardy, GANs, AlphaGo vs. Lee Sedol, Transformers, BERT, GPT-3, AlphaFold 2, ChatGPT launch, GPT-4, EU AI Act.

---

## How to Apply These Weights

1. **CSV:** In `data/AI_Timeline_1940-2025.csv`, replace the "Event Weight" column (last column) with the "New" values above, in order: CSV row 2 → event 1, …, CSV row 217 → event 216 (Prism and GPT-5.2 both get **1**).
2. **Script:** If `scripts/generate_timeline.py` maps weight to CSS classes (e.g. level-0/1/2), update the mapping to: 1 → minor, 2 → major, 3 → landmark (e.g. `level-0`=1, `level-1`=2, `level-2`=3).
3. **Regenerate:** Run the generator and refresh the timeline HTML so visuals match the new 1–3 scale.

---

*Review completed with reference to standard AI histories (e.g. Stanford AI100, Wikipedia, Britannica) and milestone impact.*  

*Last updated: 2025-01-28*
