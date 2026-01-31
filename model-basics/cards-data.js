// Model Basics - Card Data
// All content for the topic cards

const cardsData = [
    {
        category: 'arch',
        badge: 'Overview',
        title: 'The "Frozen" Pipeline',
        description: 'AI models are sophisticated mathematical engines—not conscious beings, but carefully trained statistical systems.',
        paragraphs: [
            'At its core, an AI model is a <strong>frozen mathematical pipeline</strong> containing billions of fixed numbers (called weights) arranged in layers. Once training is complete, these numbers are locked in place and don\'t change when you chat with the model.',
            'Think of it like a printed book: the content is fixed after publication. You can read it, annotate the margins, or discuss it with others—but the text itself remains unchanged. Similarly, chatting with AI doesn\'t update its "brain."'
        ],
        bullets: [
            'Understanding this architecture helps predict failures, hallucinations, and informs effective use of tools like RAG',
            'The pipeline flows: <strong>Architecture</strong> → <strong>Training</strong> → <strong>Inference</strong> → <strong>Advanced Capabilities</strong>'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Critical Distinction:</strong> Training updates the model\'s "brain" (weights). Inference runs data through the frozen brain. Chatting with AI doesn\'t teach it anything long-term—it only affects the current conversation.'
        }
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: 'The Blueprint',
        description: 'Modern AI is built on the Transformer architecture—a neural network design that revolutionized language understanding.',
        paragraphs: [
            'The <strong>Transformer</strong> is the foundational architecture powering GPT, Claude, Gemini, and most modern AI. It processes text through layers of interconnected nodes, where each layer learns increasingly abstract patterns.',
            'Early layers detect simple patterns (punctuation, word endings), while deeper layers grasp complex concepts (sarcasm, logical reasoning, thematic connections). The model\'s size is measured in <strong>parameters</strong>—billions of adjustable weights that encode learned knowledge.'
        ],
        bullets: [
            '<strong>Parameters:</strong> Billions of adjustable weights (GPT-4: hundreds of billions, some models exceed 1 trillion)',
            '<strong>Context Window:</strong> The model\'s "working memory" for a conversation (ranges from 4K to 200K+ tokens)',
            '<strong>Attention Mechanism:</strong> A spotlight that focuses on relevant words when processing each new token'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>Analogy:</strong> Attention works like reading with a highlighter—as you process each word, you mentally highlight related words earlier in the text to understand relationships and meaning.'
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
            'Training data cutoff means models lack knowledge of events after training (e.g., Claude\'s cutoff is April 2024)',
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
            '<strong>1. System Prompt:</strong> Hidden instructions defining the model\'s persona and behavior ("You are Claude, a helpful AI assistant…").',
            '<strong>2. Conversation History:</strong> Every prior message in the conversation is re-sent on each turn, consuming context window space.',
            '<strong>3. User Prompt:</strong> Your actual message, often wrapped in XML tags for parsing (<code>&lt;user_query&gt;</code>).'
        ],
        bullets: [
            'Long conversations get exponentially slower—the entire history is reprocessed every turn',
            'At context limit, early messages are dropped (the model "forgets" the start of long chats)',
            'Prompt engineering exploits this structure: clear instructions, examples, and formatting improve outputs'
        ],
        callout: {
            type: 'note',
            content: '<strong>Note:</strong> This is why long conversations become expensive and slow. Each new response requires reprocessing thousands of prior tokens. Some systems use caching or summarization to mitigate this.'
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
        category: 'adv',
        badge: 'Advanced',
        title: 'RAG (Retrieval-Augmented Generation)',
        description: 'RAG combats hallucinations and knowledge cutoffs by injecting external documents directly into the model\'s context.',
        paragraphs: [
            'Models have <strong>knowledge cutoffs</strong> (e.g., April 2024) and no access to private documents. <strong>Retrieval-Augmented Generation (RAG)</strong> solves this by dynamically fetching relevant information and inserting it into the prompt.',
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
            content: '<strong>Note:</strong> Modern models use an internal "scratchpad" (hidden reasoning tokens) to plan which tools to call and in what order. This enables complex workflows without human intervention.'
        }
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: 'Reasoning: Two Paradigms',
        description: 'Reasoning capability comes from two distinct approaches: prompting techniques and dedicated inference-time compute.',
        paragraphs: [
            '<strong>Chain of Thought (CoT):</strong> A prompting technique where you ask the model to "think step by step." This encourages intermediate reasoning, improving accuracy on math and logic tasks. It\'s a prompt hack, not a model feature.',
            '<strong>Inference-Time Compute:</strong> Models like OpenAI\'s o1, o3, and DeepSeek-R1 generate <strong>hidden reasoning tokens</strong> before answering. They "think longer" by exploring multiple solution paths internally, trading speed for accuracy. This is baked into the model.'
        ],
        bullets: [
            'CoT: Explicitly included in the prompt ("Let\'s solve this step by step")',
            'Inference-time compute: Model automatically spends extra computation reasoning privately',
            'Modern reasoning models can use thousands of hidden tokens to solve hard problems'
        ],
        callout: {
            type: 'insight',
            content: '<strong>The Shift:</strong> Traditional CoT is a user-side prompting trick. Modern reasoning models (o1, R1) embed deliberate thinking into the architecture—spending compute during inference to reduce errors without additional training.'
        },
        resources: [
            { icon: '🎬', title: 'Chain-of-Thought Explained', meta: '8 min', url: 'https://www.youtube.com/watch?v=AFE6x81AP4k' },
            { icon: '📺', title: 'Test-Time Scaling', meta: '12 min • DeepSeek-R1 & o1', url: 'https://www.youtube.com/watch?v=NbE8MoR8mPw' }
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
        category: 'adv',
        badge: 'Advanced',
        title: 'Embeddings',
        description: 'Embeddings convert text into high-dimensional vectors, enabling semantic search and similarity comparisons.',
        paragraphs: [
            '<strong>Embeddings</strong> are dense numerical representations of text. A sentence becomes a vector of 768-12,288 numbers in a high-dimensional space where semantic similarity = geometric proximity.',
            'This enables powerful capabilities: "King - Man + Woman ≈ Queen" (vector arithmetic captures relationships), semantic search (find documents by meaning, not keywords), clustering (group similar items), recommendations.'
        ],
        bullets: [
            'Underpins RAG: user queries and documents both converted to embeddings for matching',
            'Multimodal embeddings: images, audio, and text can share the same vector space',
            'Modern models produce embeddings as intermediate layer outputs (not trained separately)'
        ],
        callout: {
            type: 'note',
            content: '<strong>Insight:</strong> Embeddings are why models understand synonyms, analogies, and context. "Puppy" and "dog" occupy nearby points in embedding space—the model "knows" they\'re related without explicit rules.'
        },
        resources: [
            { icon: '🎬', title: 'Embeddings Explained', meta: '18 min • 3D visualizations', url: 'https://www.youtube.com/watch?v=eUbKYEC0D3Y' }
        ]
    },
    {
        category: 'infer',
        badge: 'Summary',
        title: 'The Complete Pipeline',
        description: 'Here\'s how everything fits together when you send a message to an AI model.',
        paragraphs: [
            'The full inference pipeline combines all these concepts into a coordinated system:'
        ],
        bullets: [
            '<strong>1. Input Assembly:</strong> System prompt + conversation history + user message → tokenized into integer IDs',
            '<strong>2. Forward Pass:</strong> Token IDs flow through the frozen Transformer layers (attention, feed-forward, layer normalization)',
            '<strong>3. Sampling:</strong> Output layer produces logits (raw scores) → softmax → probability distribution → sample next token (controlled by temperature)',
            '<strong>4. Autoregressive Loop:</strong> Append the generated token, feed it back in, repeat until hitting a stop token or length limit',
            '<strong>5. Post-Processing:</strong> Apply safety filters, format citations, execute tool calls if requested',
            '<strong>6. Streaming:</strong> Tokens sent to user incrementally as they\'re generated (creates illusion of "typing")'
        ],
        callout: {
            type: 'note',
            content: '<strong>Note:</strong> This entire process—tokenization, forward pass, sampling, detokenization—takes milliseconds to seconds per token, depending on model size, hardware, and reasoning depth.'
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