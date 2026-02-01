// Model Basics - Card Data
// All content for the topic cards

const cardsData = [
    {
        category: 'arch',
        badge: 'Overview',
        title: 'How modern AI models work',
        description: 'AI models are sophisticated mathematical engines—not conscious beings, but carefully trained statistical systems.',
        paragraphs: [
            'In this presentation, we are going to demystify how modern AI systems work. Today, large models are descendants of the perceptrons and neural networks in our timeline—scaled up, trained on massive datasets, and organized into architectures like the <strong>Transformer</strong>.',
            'We will look at how the models are built and trained, and then what happens during inference when a user makes a request. Along the way, we will see why outputs can feel human, where the system runs aground (hallucinations, missing context, brittle logic), and why running these models—and the tools around them—can be complex and expensive.'
        ],
        bullets: [
            'Understanding this architecture helps predict failures, hallucinations, and informs effective use of tools like RAG',
            'The pipeline flows: <strong>Architecture</strong> → <strong>Training</strong> → <strong>Inference</strong> → <strong>Advanced Capabilities</strong>'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Training vs. Inference:</strong> Training updates the model\'s "brain" (weights)—a massive structure that can contain trillions of parameters. Inference is the act of querying that "frozen" brain. Chatting provides temporary context, but it does not permanently teach the model or update its knowledge base.'
        },
        resources: [
            { icon: '📺', title: 'Generative AI in a Nutshell - how to survive and thrive in the age of AI', meta: '18 min • Henrik Knibbe', url: 'https://www.youtube.com/watch?v=2IK3DFHRFfw' },
            { icon: '🌐', title: 'OKAI — An Interactive Introduction to Artificial Intelligence', meta: 'Interactive site • Brown University', url: 'https://okai.brown.edu/' },
            { icon: '📺', title: 'Large Language Models explained briefly', meta: '8 min • 3Blue1Brown', url: 'https://www.youtube.com/watch?v=LPZh9BOjkQs' }
        ]
    },
    {
        category: 'arch',
        badge: 'History',
        title: 'Before Transformers',
        description: 'Understanding why the Transformer architecture was revolutionary requires looking at what came before.',
        paragraphs: [
            'Early neural networks for language processing used <strong>Recurrent Neural Networks (RNNs)</strong> and <strong>Long Short-Term Memory (LSTM)</strong> architectures. These models processed text sequentially—one word at a time, in order—maintaining a "hidden state" that tried to remember earlier context.',
            'The problem? Sequential processing was slow (couldn\'t parallelize across GPUs), and models struggled with long-range dependencies due to <strong>vanishing gradients</strong>. By the time an RNN reached word 50, it had largely "forgotten" word 1.',
            'The 2017 paper "Attention Is All You Need" introduced the <strong>Transformer</strong>, which eliminated sequential processing entirely. Instead of reading word-by-word, it processes all tokens simultaneously using the attention mechanism—enabling parallelization and capturing dependencies across arbitrary distances.'
        ],
        bullets: [
            '<strong>RNNs/LSTMs:</strong> Sequential processing (slow), vanishing gradients (poor long-term memory)',
            '<strong>CNNs:</strong> Worked for images but struggled with variable-length text and long dependencies',
            '<strong>The Breakthrough:</strong> Attention mechanism allows every token to directly "look at" every other token in parallel'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Why This Matters:</strong> The shift from sequential to parallel processing is why modern AI could scale to billions of parameters and trillion-token datasets. RNNs couldn\'t scale effectively—Transformers could.'
        }
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: 'The Transformer Architecture',
        description: 'Modern AI is built on the Transformer—a neural network design that processes all tokens in parallel using layers of attention and computation.',
        paragraphs: [
            'The <strong>Transformer</strong> is the foundational architecture powering GPT, Claude, Gemini, and most modern AI. It processes text through layers of interconnected nodes, where each layer learns increasingly abstract patterns.',
            'Each Transformer layer has two main components: <strong>Attention</strong> (relating tokens to each other) and <strong>Feed-Forward Networks</strong> (processing individual tokens through non-linear transformations). These alternate in a repeating pattern, with normalization and residual connections stabilizing training.',
            'Early layers detect simple patterns (punctuation, word endings), while deeper layers grasp complex concepts (sarcasm, logical reasoning, thematic connections). The model\'s size is measured in <strong>parameters</strong>—billions of adjustable weights that encode learned knowledge.'
        ],
        bullets: [
            '<strong>Parameters:</strong> Adjustable weights that store learned patterns (model sizes range from millions to hundreds of billions+)',
            '<strong>Layers:</strong> Each layer alternates between attention (relating tokens) and feed-forward computation (processing tokens)',
            '<strong>Positional Encodings:</strong> Added to each token to preserve word order—without them, "dog bites man" and "man bites dog" would be indistinguishable',
            '<strong>Context Window:</strong> The model\'s "working memory" for a conversation (ranges from 4K to 200K+ tokens)'
        ],
        callout: {
            type: 'note',
            content: '<strong>Layer Structure:</strong> Think of each layer as a two-step process: (1) Attention asks "which other tokens matter for understanding this one?" (2) Feed-forward networks apply complex transformations to extract features. Repeat this 12-96 times depending on model size.'
        },
        resources: [
            { icon: '📺', title: 'A Student\'s Guide to Vectors and Tensors', meta: '12 min • Dan Fleisch', url: 'https://www.youtube.com/watch?v=f5liqUk0ZTw' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: 'How Attention Works',
        description: 'Attention is the core mechanism that allows Transformers to understand relationships between words, no matter how far apart they are.',
        paragraphs: [
            'When processing each token, the model needs to decide which other tokens in the sequence are relevant. <strong>Attention</strong> solves this through three learned representations for every token:',
            'The model computes attention scores by comparing the Query of the current token against the Keys of all previous tokens (including itself). High scores mean "these tokens are relevant to understanding this one." These scores create a weighted average of the Values—tokens with high attention get more weight.'
        ],
        bullets: [
            '<strong>Query (Q):</strong> "What am I looking for?" Each token generates a query vector representing what information it needs',
            '<strong>Key (K):</strong> "What do I offer?" Each token generates a key vector advertising its content',
            '<strong>Value (V):</strong> "Here\'s my actual information." Each token generates a value vector containing its semantic content',
            '<strong>Self-Attention:</strong> Each token attends to all tokens in the sequence (including itself)',
            '<strong>Multi-Head Attention:</strong> Multiple attention mechanisms run in parallel, each learning different relationships (syntax, semantics, coreference)',
            '<strong>Attention Scores:</strong> Determine which tokens influence each other—visualizing these reveals what the model "focuses on"',
            '<strong>Context Window Limit:</strong> Attention requires comparing every token to every other token—cost grows quadratically with length'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>Analogy:</strong> Imagine reading a sentence where each word can ask questions to all previous words. "Loves" asks "who?" and attends strongly to "Sarah." "Bank" asks "context?" and attends to nearby words to distinguish "river bank" from "money bank." Attention automates this process across thousands of tokens simultaneously.'
        },
        resources: [
            { icon: '🎬', title: 'Attention Is All You Need', meta: '15 min • Visual walkthrough', url: 'https://www.youtube.com/watch?v=wjZofJX0v4M' },
            { icon: '📺', title: '3Blue1Brown Attention', meta: '26 min • Animated explanation', url: 'https://www.youtube.com/watch?v=eMlx5fFNoYc' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: 'Tokenization',
        description: 'Models don\'t understand text directly—they process numeric tokens that represent pieces of words.',
        paragraphs: [
            'AI models operate on numbers, not letters. A <strong>tokenizer</strong> converts text into integer IDs representing vocabulary fragments. For example, "Ingenious" might split into three tokens: <code>In</code>, <code>gen</code>, and <code>ious</code>.',
            'This approach, called <strong>Byte-Pair Encoding (BPE)</strong>, balances efficiency and flexibility. Common words stay whole ("the"), while rare words split into recognizable parts. Multimodal models extend this: images become "patches," audio becomes tokens.'
        ],
        bullets: [
            'Token count determines cost and speed—more tokens = higher compute',
            'Tokenization explains quirks: why models struggle with spelling backward (tokens don\'t map 1:1 to letters)',
            '<strong>Everything is next-token prediction:</strong> Poetry, code, math—all reduced to "what token comes next?"'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Token Prediction Paradigm:</strong> This single task—predicting the next token—enables every capability. It\'s why models excel at pattern completion but may fail at precise arithmetic (it\'s token prediction, not calculation).'
        },
        resources: [
            { icon: '🛠️', title: 'OpenAI Tokenizer', meta: 'Interactive tool', url: 'https://platform.openai.com/tokenizer' },
            { icon: '📺', title: 'Build GPT Tokenizer', meta: '2h 13min • Andrej Karpathy', url: 'https://www.youtube.com/watch?v=zduSFxRajkE' }
        ]
    },
    {
        category: 'train',
        badge: 'Training',
        title: 'Pre-Training',
        description: 'Pre-training is where models learn the patterns, facts, and structures of human knowledge from massive text datasets.',
        paragraphs: [
            'During pre-training, the model consumes <strong>trillions of tokens</strong> from books, websites, research papers, and code repositories. The training objective is simple: predict the next token. Wrong predictions trigger tiny weight adjustments via backpropagation.',
            'This process takes months on thousands of GPUs and costs millions of dollars. The result? A <strong>base model</strong> that can complete sentences, generate code, and recall facts—but often produces rambling or unhelpful outputs.'
        ],
        bullets: [
            '<strong>Scaling Laws:</strong> Predictable relationship: more parameters + more data + more compute = better performance',
            'Training data cutoff means models may lack knowledge of events after training (the cutoff varies by model)',
            'Base models understand language structure but haven\'t learned to be helpful assistants yet'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>Analogy:</strong> Pre-training is like teaching a child to predict the next word in stories. They learn grammar, vocabulary, and facts—but not how to hold a conversation or follow instructions helpfully.'
        },
        resources: [
            { icon: '📺', title: 'Neural Networks & Backprop', meta: '2+ hrs • Andrej Karpathy', url: 'https://www.youtube.com/watch?v=VMj-3S1tku0' },
            { icon: '🎬', title: 'What is Backpropagation?', meta: '14 min • 3Blue1Brown', url: 'https://www.3blue1brown.com/lessons/backpropagation' }
        ]
    },
    {
        category: 'train',
        badge: 'Training',
        title: 'Post-Training',
        description: 'Post-training transforms a knowledgeable but unruly base model into a helpful, safe, and aligned assistant.',
        paragraphs: [
            'Base models know a lot but behave poorly—generating offensive content, refusing simple requests, or rambling endlessly. <strong>Post-training</strong> teaches them to be useful assistants through two key techniques:',
            '<strong>Supervised Fine-Tuning (SFT):</strong> Humans write ideal responses to thousands of prompts. The model learns to mimic this helpful behavior.',
            '<strong>Reinforcement Learning from Human Feedback (RLHF):</strong> Humans rank multiple model responses (A > B > C). The model learns to maximize preference scores. Modern alternatives like <strong>DPO</strong> (Direct Preference Optimization) streamline this process.'
        ],
        bullets: [
            'Post-training instills safety guardrails (refusing harmful requests)',
            'Models learn conversational norms: being concise, admitting uncertainty, citing sources',
            'Trade-off: alignment reduces raw creativity and capability slightly'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Safety vs. Capability:</strong> Post-training trades some raw capability for alignment. An aligned model might refuse edge-case requests a base model would attempt—prioritizing safety over unbounded helpfulness.'
        },
        resources: [
            { icon: '📺', title: 'RLHF, Clearly Explained', meta: '18 min • StatQuest', url: 'https://www.youtube.com/watch?v=qPN_XZcJf_s' },
            { icon: '🎬', title: 'RLHF in 4 Minutes', meta: '4 min • Sebastian Raschka', url: 'https://www.youtube.com/watch?v=vJ4SsfmeQlk' }
        ]
    },
    {
        category: 'train',
        badge: 'Training',
        title: 'Bias, Fairness & Limitations',
        description: 'AI models inherit the biases, gaps, and perspectives present in their training data—they are mirrors, not arbiters of truth.',
        paragraphs: [
            'Training data comes from the internet, books, and human-generated content—all of which contain biases, stereotypes, and uneven representation. Models learn these patterns just as they learn grammar and facts. If training data overrepresents certain demographics or perspectives, the model will too.',
            '<strong>Post-training alignment</strong> can reduce some harmful outputs (e.g., refusing to generate hate speech), but it doesn\'t eliminate underlying biases. A model might still generate biased resume summaries, make assumptions based on names, or reflect cultural stereotypes—even when trying to be helpful.',
            'This matters in high-stakes domains: healthcare (misdiagnosis patterns), hiring (resume screening bias), legal systems (risk assessment), education (unequal tutoring quality). No model is "objective"—all reflect their training data\'s worldview.'
        ],
        bullets: [
            '<strong>Sources of Bias:</strong> Training data imbalances, historical stereotypes, language and cultural gaps, majority perspectives dominating',
            '<strong>Types of Harm:</strong> Stereotyping, erasure (underrepresented groups), performance gaps (works better for some demographics)',
            '<strong>Mitigation Strategies:</strong> Diverse training data, red-teaming for harmful outputs, constitutional AI principles, ongoing monitoring',
            '<strong>User Responsibility:</strong> Critical evaluation of outputs, awareness of limitations, human oversight in high-stakes decisions'
        ],
        callout: {
            type: 'insight',
            content: '<strong>No Silver Bullet:</strong> Bias mitigation is an ongoing process, not a solved problem. Even the most carefully trained models can produce biased outputs. The goal is harm reduction and transparency, not perfection. Always apply human judgment, especially in consequential decisions.'
        },
        resources: [
            { icon: '📺', title: 'AI Bias Explained', meta: '9 min • TEDx', url: 'https://www.youtube.com/watch?v=59bMh59JQDo' }
        ]
    },
    {
        category: 'infer',
        badge: 'Inference',
        title: 'The Frozen State',
        description: 'After training, model weights are frozen—inference runs data through this fixed architecture without learning.',
        paragraphs: [
            'Once training completes, the model\'s weights are <strong>locked</strong>. Inference (generating responses) reads these weights but never modifies them. This is why chatting doesn\'t teach the model anything permanent—corrections only affect the current conversation\'s context.',
            '<strong>In-Context Learning</strong> is the workaround: if you provide examples in your prompt, the model adapts its outputs to match those patterns—but only for that single interaction. After the conversation ends, the model "forgets" everything.'
        ],
        bullets: [
            'No learning during inference: feedback doesn\'t update weights',
            'Context window is the only "memory"—once it\'s full, earlier messages get truncated',
            'In-context learning mimics adaptation without changing the underlying model'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>Analogy:</strong> Training engraves knowledge into stone tablets. Inference reads the tablets but can\'t modify them. Taking notes during a conversation doesn\'t change the stone\'s engraving.'
        }
    },
    {
        category: 'infer',
        badge: 'Inference',
        title: 'The Prompt Stack',
        description: 'Every model interaction involves a carefully structured stack of hidden instructions, history, and user input.',
        paragraphs: [
            'When you send a message, the model doesn\'t just see your text. It processes a <strong>prompt stack</strong> with multiple layers:',
            '<strong>1. System Prompt:</strong> Hidden instructions defining the model\'s persona and behavior ("You are a helpful AI assistant…").',
            '<strong>2. Conversation History:</strong> Every prior message in the conversation is re-sent on each turn, consuming context window space.',
            '<strong>3. User Prompt:</strong> Your actual message, often wrapped in XML tags for parsing (<code>&lt;user_query&gt;</code>).'
        ],
        bullets: [
            'Long conversations get slower and more expensive—the entire history is typically reprocessed every turn',
            'At context limit, early messages are dropped (the model "forgets" the start of long chats)',
            '<strong>Context rot:</strong> In long chats, constraints can get buried or dropped—leading to drift and contradictions',
            'Prompt engineering exploits this structure: clear instructions, examples, and formatting improve outputs'
        ],
        callout: {
            type: 'note',
            content: '<strong>Note:</strong> This is why long conversations become expensive and slow. Each new response requires reprocessing thousands of prior tokens. <strong>Tip:</strong> Periodically restate the goal and key constraints (or start a fresh thread with a short summary) to reduce drift.'
        }
    },
    {
        category: 'infer',
        badge: 'Inference',
        title: 'Probabilities & Hallucinations',
        description: 'Models generate text by sampling from probability distributions—a process that enables creativity but allows errors.',
        paragraphs: [
            'For every token, the model outputs a <strong>probability distribution</strong> across its entire vocabulary (100K+ tokens). A <strong>temperature</strong> setting controls sampling:',
            '<strong>Temperature 0:</strong> Always pick the highest-probability token (deterministic, conservative). <strong>Temperature 0.7-1.0:</strong> Sample probabilistically, enabling creative variation but risking drift off-topic.',
            '<strong>Hallucinations</strong> occur when the model generates fluent but false text. Why? The model prioritizes "statistically plausible" over "factually accurate"—it lacks a truth-checking mechanism.'
        ],
        bullets: [
            'Higher temperature = more creative but less predictable',
            'Hallucinations aren\'t "lying"—the model doesn\'t know it\'s wrong',
            'Low-data domains (recent events, niche topics) are hallucination-prone'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Why Hallucinations Persist:</strong> No ground-truth verification exists during inference. The model\'s objective is "generate plausible text," not "generate true text." Hallucinations are a fundamental limitation of the next-token prediction paradigm.'
        }
    },
    {
        category: 'infer',
        badge: 'Summary',
        title: 'What happens when you send a message?',
        description: 'A simple end-to-end view of the inference loop.',
        paragraphs: [
            'Every chat turn runs the same basic pipeline: assemble the prompt, run a forward pass, pick the next token, and repeat.'
        ],
        bullets: [
            '<strong>1. Input Assembly:</strong> System prompt + conversation history + user message → tokenized into integer IDs',
            '<strong>2. Forward Pass:</strong> Token IDs flow through the frozen Transformer layers (attention + feed-forward layers)',
            '<strong>3. Token Selection:</strong> The model scores possible next tokens → picks one (greedy or sampling/temperature)',
            '<strong>4. Autoregressive Loop:</strong> Append the generated token, feed it back in, repeat until a stop condition',
            '<strong>5. Optional System Steps:</strong> Apply safety policies, execute tool calls, and/or insert retrieved documents (RAG)',
            '<strong>6. Streaming:</strong> Tokens are sent to the user incrementally as they are generated'
        ],
        callout: {
            type: 'note',
            content: '<strong>Note:</strong> Speed depends on model size, hardware, context length, and how much extra reasoning/tool use is happening.'
        }
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: 'Embeddings',
        description: 'Embeddings convert text into vectors, enabling semantic search and similarity comparisons.',
        paragraphs: [
            '<strong>Embeddings</strong> are dense numerical representations of content. A sentence becomes a vector (a list of numbers) in a space where semantic similarity ≈ geometric proximity.',
            'This enables semantic search (find documents by meaning, not keywords), clustering (group similar items), and recommendations. It is also a core ingredient in most RAG systems.'
        ],
        bullets: [
            'Underpins RAG: user queries and documents are both converted to embeddings for matching',
            'Multimodal embeddings: images, audio, and text can share the same vector space',
            'Embeddings can come from dedicated embedding models or from intermediate layers of larger models'
        ],
        callout: {
            type: 'note',
            content: '<strong>Insight:</strong> Embeddings are why models understand synonyms, analogies, and context. "Puppy" and "dog" occupy nearby points in embedding space—the system "knows" they\'re related without explicit rules.'
        },
        resources: [
            { icon: '🎬', title: 'Embeddings Explained', meta: '18 min • 3D visualizations', url: 'https://www.youtube.com/watch?v=eUbKYEC0D3Y' }
        ]
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: 'Multimodal Models',
        description: 'Modern AI can process and generate not just text, but images, audio, video—all converted into tokens and embeddings.',
        paragraphs: [
            '<strong>Multimodal models</strong> extend the token-prediction paradigm beyond text. Images are split into patches (like a grid), each patch encoded as a token by a vision encoder. Audio waveforms are converted to spectrograms, then tokenized. Video combines both approaches frame-by-frame.',
            'The key innovation: all modalities are projected into a <strong>unified embedding space</strong>. A picture of a cat and the word "cat" occupy nearby points in this space—the model "knows" they\'re related. This enables cross-modal reasoning: describe what\'s in an image, generate images from text descriptions, transcribe and translate audio.',
            'Examples: <strong>GPT-4V</strong> (vision + text), <strong>Gemini</strong> (text + images + video), <strong>DALL-E/Midjourney</strong> (text → images), <strong>Whisper</strong> (audio → text). The same Transformer architecture and attention mechanisms work across all modalities.'
        ],
        bullets: [
            '<strong>Vision Tokenization:</strong> Images split into 16×16 or 32×32 patches, each patch becomes a token vector',
            '<strong>Audio Tokenization:</strong> Waveforms → spectrograms (frequency over time) → token sequences',
            '<strong>Cross-Modal Attention:</strong> Text tokens can attend to image patches and vice versa',
            '<strong>Why It Matters:</strong> Enables richer interactions (ask questions about photos), creative tools (AI art generation), accessibility (image descriptions for vision impairment)'
        ],
        callout: {
            type: 'note',
            content: '<strong>Unified Architecture:</strong> The same core Transformer that processes text can process images and audio—only the tokenization step differs. This is why multimodal capabilities emerged quickly: the architecture was already designed to handle arbitrary token sequences.'
        },
        resources: [
            { icon: '🎬', title: 'How Multimodal Models Work', meta: '12 min • Visual explanation', url: 'https://www.youtube.com/watch?v=vAmKB7iPkWw' }
        ]
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: 'RAG (Retrieval-Augmented Generation)',
        description: 'RAG combats hallucinations and knowledge cutoffs by injecting external documents directly into the model\'s context.',
        paragraphs: [
            'Models have <strong>knowledge cutoffs</strong> and no access to your private documents. <strong>Retrieval-Augmented Generation (RAG)</strong> solves this by dynamically fetching relevant information and inserting it into the prompt.',
            'The process: (1) User asks a question. (2) Convert question to an <strong>embedding</strong> (vector). (3) Search a vector database for similar documents. (4) Paste retrieved docs into the model\'s context. (5) Model answers using both its training and the retrieved text.'
        ],
        bullets: [
            'Reduces hallucinations by grounding responses in real documents',
            'Enables citations (model can reference specific sources)',
            'Keeps models current without expensive retraining'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>Analogy:</strong> RAG is like allowing a student to bring textbooks into an exam. They still use reasoning and comprehension—but can look up specific facts instead of guessing from memory.'
        },
        resources: [
            { icon: '🎬', title: 'What is RAG?', meta: '6 min • IBM', url: 'https://youtube.com/watch?v=T-D1OfcDW1M' }
        ]
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: 'Tool Use (Function Calling)',
        description: 'Models can\'t execute code or perform calculations internally—tool use lets them request external actions.',
        paragraphs: [
            'Large language models are terrible at precise math (remember: they\'re next-token predictors, not calculators). <strong>Tool use</strong> (also called function calling) provides a workaround:',
            'The model outputs a structured request: <code>{"tool": "calculator", "input": "25*48"}</code>. Your system executes the tool and feeds the result back. The model continues generating, now informed by accurate computation.'
        ],
        bullets: [
            'Enables: web search, database queries, code execution, sending emails, controlling robotics',
            'Models can chain multiple tools sequentially (search web → extract data → calculate → format response)',
            'Most capable models support multi-step tool orchestration internally'
        ],
        callout: {
            type: 'note',
            content: '<strong>Note:</strong> Many systems use an internal "scratchpad" (hidden reasoning tokens) to plan which tools to call and in what order. This enables complex workflows without human intervention.'
        }
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: 'Reasoning: Two Paradigms',
        description: 'Reasoning capability comes from two distinct approaches: prompting techniques and dedicated inference-time compute.',
        paragraphs: [
            '<strong>Chain of Thought (CoT):</strong> A prompting technique where you ask the model to "think step by step." This encourages intermediate reasoning, improving accuracy on math and logic tasks. It\'s a prompt hack, not a model feature.',
            '<strong>Inference-Time Compute:</strong> Some reasoning-focused models generate <strong>hidden reasoning tokens</strong> before answering. They "think longer" by exploring multiple solution paths internally, trading speed for accuracy. This behavior is baked into the model.'
        ],
        bullets: [
            'CoT: Explicitly included in the prompt ("Let\'s solve this step by step")',
            'Inference-time compute: Model automatically spends extra computation reasoning privately',
            'Modern reasoning models can use thousands of hidden tokens to solve hard problems'
        ],
        callout: {
            type: 'insight',
            content: '<strong>The Shift:</strong> Traditional CoT is a user-side prompting trick. Modern reasoning-focused models embed deliberate thinking into the system—spending extra compute during inference to reduce errors without additional training.'
        },
        resources: [
            { icon: '🎬', title: 'Chain-of-Thought Explained', meta: '8 min', url: 'https://www.youtube.com/watch?v=AFE6x81AP4k' },
            { icon: '📺', title: 'Test-Time Scaling', meta: '12 min • Reasoning at inference time', url: 'https://www.youtube.com/watch?v=NbE8MoR8mPw' }
        ]
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: 'Agentic Workflows',
        description: 'Combining reasoning, tools, and planning creates autonomous agents that can accomplish complex multi-step tasks.',
        paragraphs: [
            'An <strong>agent</strong> is an AI system that can perceive, reason, plan, and act autonomously. By combining inference-time reasoning with tool use, agents break down complex goals into actionable steps.',
            '<strong>ReAct Pattern:</strong> Thought → Action → Observation → Next Thought. The model iteratively reasons, uses tools, observes results, and adjusts its plan.',
            'Examples: Booking a flight (search → check calendar → compare prices → confirm), debugging code (run → read error → fix → re-run), conducting research (search → summarize → synthesize).'
        ],
        bullets: [
            '<strong>Planning:</strong> Decompose "book a trip" into searchable sub-tasks',
            '<strong>Reflection:</strong> Verify outputs ("Does this code compile?" → execute → fix → retry)',
            '<strong>Memory:</strong> Maintain state across multiple interactions (session history, external databases)'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>Analogy:</strong> A standard LLM is a smart person. An agent is that person with a computer, calculator, notepad, and the ability to search the internet—empowered to take action, not just think.'
        }
    },
    {
        category: 'infer',
        badge: 'Conclusion',
        title: 'Understanding the System',
        description: 'AI models are not conscious—they\'re sophisticated statistical systems that mirror human knowledge.',
        paragraphs: [
            'Modern AI isn\'t magic. It\'s a <strong>high-fidelity statistical mirror</strong> of human-created text, trained on trillions of tokens to predict plausible continuations. Understanding the frozen pipeline, tokenization, training phases, and inference mechanics demystifies both capabilities and limitations.',
            'This knowledge empowers you to use AI more effectively: anticipate hallucinations in low-data domains, design better prompts leveraging in-context learning, choose appropriate tools (RAG for facts, calculators for math), and recognize the difference between pattern completion and genuine reasoning.'
        ],
        bullets: [
            'Models excel at: language fluency, pattern matching, creative generation, code completion',
            'Models struggle with: precise arithmetic, recent events (post-cutoff), multi-step planning without reasoning models, verifying factual accuracy',
            'The future: larger context windows, multimodal fusion, better reasoning, improved tool orchestration'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Final Thought:</strong> The "magic" of AI isn\'t that it thinks—it\'s that billions of mathematical operations, trained on trillions of tokens, compress human knowledge into a reusable, frozen artifact. Understanding this transforms you from a passive user into an informed practitioner.'
        }
    }
];