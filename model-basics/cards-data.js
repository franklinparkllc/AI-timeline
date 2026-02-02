// Model Basics - Card Data
// All content for the topic cards

const cardsData = [
    {
        category: 'arch',
        badge: 'Overview',
        title: 'Introduction to modern AI models',
        description: 'AI models are sophisticated mathematical engines that have seen remarkable growth in the last decade',
        paragraphs: [
            'In this presentation, we are going to demystify how modern AI systems work.',
            'It is useful to think about model architecture, training, and inference as separate stages in a pipeline.',
            'At the heart of modern models is a concept called the <strong>Transformer</strong>, which is a type of neural network architecture that is designed to process text data. Transformers were defined in a seminal paper in 2017 by Vaswani et al. and have since become the de facto standard for language model architecture. Transformers unlocked the ability to train models with billions of parameters, which is what allows modern models to be so powerful.'
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
            'Early neural networks for language processing used <strong>Recurrent Neural Networks (RNNs)</strong> and <strong>Long Short-Term Memory (LSTM)</strong> architectures. These models processed text sequentially—one word at a time, in order—maintaining and keeping the context of previous words was a resource intensive process.',
            'The problem? Sequential processing was slow (couldn\'t parallelize across GPUs), and models struggled with long-range dependencies. By the time an RNN reached word 50, it had largely "forgotten" word 1.',
            '<strong>Seq2Seq</strong> (encoder-decoder) models added <strong>attention</strong>—first for machine translation—so the decoder could "look at" relevant parts of the input. Attention helped, but the backbone was still sequential RNNs. The precursors started appearing around 2014, with the first successful implementation of attention in 2015 by Bahdanau et al. and Vaswani et al. in 2017.',
            'The 2017 paper "Attention Is All You Need" introduced the <strong>Transformer</strong>, which dropped the RNN entirely. It processes all tokens in parallel using only attention—enabling the scale we see today.'
        ],
        bullets: [
            '<strong>RNNs/LSTMs:</strong> Sequential processing (slow), vanishing gradients (poor long-term memory)',
            '<strong>Seq2Seq:</strong> Encoder-decoder + attention (e.g. for translation); attention helped, but RNNs remained the bottleneck',
            '<strong>CNNs:</strong> Worked for images but struggled with variable-length text and long dependencies',
            '<strong>The Breakthrough:</strong> Transformer keeps attention, drops recurrence—every token can "look at" every other token in parallel'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Why This Matters:</strong> The shift from sequential to parallel processing is why modern AI could scale to billions of parameters and trillion-token datasets. RNNs couldn\'t scale effectively—Transformers could.'
        },
        resources: [
            { icon: '🌐', title: 'Attention? Attention!', meta: 'Lilian Weng • Understanding attention mechanisms', url: 'https://lilianweng.github.io/posts/2018-06-24-attention/' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: 'Tokens & Tokenization',
        description: 'Models don\'t read text directly—they process numeric token IDs that represent pieces of words.',
        paragraphs: [
            'AI models operate on numbers, not letters. A <strong>tokenizer</strong> converts text into integer IDs representing vocabulary fragments. For example, "Ingenious" might split into three tokens: <code>In</code>, <code>gen</code>, and <code>ious</code>.',
            'This approach (often <strong>BPE</strong> or similar) balances efficiency and flexibility: common words stay whole, rare words split into reusable parts. Multimodal models do the same idea for images (patches) and audio (chunks).'
        ],
        bullets: [
            'Token count determines cost and speed—more tokens = higher compute',
            'Tokenization explains quirks: spelling/backwards tasks are hard (tokens don\'t map 1:1 to letters)',
            '<strong>Tokens are pieces, not words:</strong> The model often sees subwords like <code>un</code> + <code>believ</code> + <code>able</code>'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Key idea:</strong> Most language models are trained on one core objective: <em>predict the next token</em>. That single skill can look like reasoning, writing, or coding—but it\'s still prediction, not guaranteed “truth” or perfect calculation.'
        },
        resources: [
            { icon: '🛠️', title: 'OpenAI Tokenizer', meta: 'Interactive tool', url: 'https://platform.openai.com/tokenizer' },
            { icon: '📺', title: 'Build GPT Tokenizer', meta: '2h 13min • Andrej Karpathy', url: 'https://www.youtube.com/watch?v=zduSFxRajkE' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: 'Embeddings, position, and context',
        description: 'After tokenization, the model turns token IDs into vectors, adds order, and works within a limited window.',
        paragraphs: [
            'A token ID is just a number. The first step is an <strong>embedding lookup</strong>: the model maps each token ID to a vector (a list of numbers) that represents meaning and usage.',
            'Because word order matters, the model adds <strong>positional information</strong> so "man bites dog" differs from "dog bites man". Then it processes the whole sequence inside a finite <strong>context window</strong> (the model’s working memory).'
        ],
        bullets: [
            '<strong>Embeddings:</strong> Vectors that represent tokens; the model updates these vectors layer by layer',
            '<strong>Positional encoding:</strong> Adds “where am I in the sequence?” so order is preserved',
            '<strong>Context window:</strong> Only tokens inside the window can influence the output',
            '<strong>Prompt budget:</strong> System + chat history + your input all share the same window'
        ],
        callout: {
            type: 'note',
            content: '<strong>Practical takeaway:</strong> When prompts get long, models may drop or compress earlier parts because they can only “see” what fits in the context window.'
        },
        resources: [
            { icon: '📺', title: 'A Student\'s Guide to Vectors and Tensors', meta: '12 min • Dan Fleisch', url: 'https://www.youtube.com/watch?v=f5liqUk0ZTw' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: 'Transformer blocks (repeat N times)',
        description: 'A Transformer is a stack of repeating blocks. Each block updates every token in two steps: mix, then refine.',
        paragraphs: [
            'Inside a block, <strong>attention</strong> lets a token pull in information from other tokens (a controlled “mixing” of context). Then a small <strong>MLP</strong> (feed-forward network) does per-token nonlinear computation to refine that token’s representation.',
            'Both steps are wrapped with <strong>residual connections</strong> (add the old signal back) and <strong>layer norm</strong> (keep values stable), which is what makes deep stacks trainable.'
        ],
        bullets: [
            '<strong>Attention = mix:</strong> each token becomes a weighted blend of other tokens’ vectors',
            '<strong>MLP = compute:</strong> transforms each token independently (adds nonlinear “feature building”)',
            '<strong>Residual:</strong> update = old + new (helps information flow through many layers)',
            '<strong>Layer norm:</strong> stabilizes training and prevents values from drifting',
            '<strong>Depth:</strong> repeating this many times builds more abstract concepts'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Why it’s easy to mix up:</strong> Attention mostly moves information between tokens; the MLP mostly transforms information within a token. Together, they let models combine context with computation.'
        },
        resources: [
            { icon: '🌐', title: 'The Illustrated Transformer', meta: 'Jay Alammar • Visual explanation', url: 'https://jalammar.github.io/illustrated-transformer/' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: 'Attention (the intuition)',
        description: 'Attention lets each token decide which other tokens matter right now.',
        paragraphs: [
            'When the model updates a token, it asks: “which other tokens should influence me?” Attention answers by assigning weights to other tokens, then mixing their information into the current token.',
            'Crucially, this happens for <em>every</em> token in parallel, producing a set of updated, context-aware token vectors.'
        ],
        bullets: [
            '<strong>Selective focus:</strong> different tokens matter for different words (e.g., resolving pronouns)',
            '<strong>Many perspectives:</strong> multiple heads learn different relationships (syntax, meaning, coreference)',
            '<strong>Causal masking (LLMs):</strong> during generation, a token can’t look “to the right” at future tokens'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>Analogy:</strong> While reading, you constantly look back to resolve meaning: “bank” checks nearby words to choose river vs money; “it” looks back for the referent. Attention automates this across thousands of tokens.'
        },
        resources: [
            { icon: '📺', title: '3Blue1Brown: Attention', meta: '26 min • Animated explanation', url: 'https://www.youtube.com/watch?v=eMlx5fFNoYc' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: 'Attention (Q, K, V)',
        description: 'Q/K/V is the recipe for turning “what should I pay attention to?” into weights and a mixed output.',
        paragraphs: [
            'For each token, the model creates three vectors: <strong>Query</strong> (what I’m looking for), <strong>Key</strong> (what I offer), and <strong>Value</strong> (my information).',
            'It scores Query vs Keys, turns those scores into weights, and takes a weighted sum of Values. That weighted sum becomes the token’s “mixed-in context.”'
        ],
        bullets: [
            '<strong>Query (Q):</strong> what this token wants to find',
            '<strong>Key (K):</strong> what this token matches on',
            '<strong>Value (V):</strong> what information this token contributes',
            '<strong>Multi-head:</strong> do this in parallel, then combine the results',
            '<strong>Cost:</strong> comparing many tokens to many tokens gets expensive as context grows'
        ],
        callout: {
            type: 'note',
            content: '<strong>One subtle point:</strong> Attention doesn’t retrieve a stored “fact” like a database query—it computes a new vector by blending existing token representations.'
        },
        resources: [
            { icon: '🎬', title: 'Attention Is All You Need', meta: '15 min • Visual walkthrough', url: 'https://www.youtube.com/watch?v=wjZofJX0v4M' }
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
        },
        resources: [
            { icon: '📘', title: 'Claude prompting best practices', meta: 'Anthropic docs • Prompt engineering', url: 'https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices' }
        ]
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
        },
        resources: [
            { icon: '📺', title: 'Why Large Language Models Hallucinate', meta: 'Video • Practical explanation', url: 'https://www.youtube.com/watch?v=cfqtFvWOfg0' },
            { icon: '🧠', title: 'Why language models hallucinate', meta: 'OpenAI • Research explainer', url: 'https://openai.com/index/why-language-models-hallucinate/' },
            { icon: '📄', title: 'Mata v. Avianca (court filing with fabricated citations)', meta: 'Primary source • SDNY docket', url: 'https://law.justia.com/cases/federal/district-courts/new-york/nysdce/1:2022cv01461/575368/54/' },
            { icon: '📰', title: 'Google Bard demo error (JWST claim)', meta: 'Reuters • Feb 2023', url: 'https://www.reuters.com/technology/google-ai-chatbot-bard-offers-inaccurate-information-company-ad-2023-02-08/' }
        ]
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
        },
        resources: [
            { icon: '📺', title: 'AI Inference: The Secret to AI\'s Superpowers', meta: 'Video • IBM Technology', url: 'https://www.youtube.com/watch?v=XtT5i0ZeHHE&t=19s' },
            { icon: '📺', title: 'An AI Prompt Engineer Shares Her Secrets', meta: 'Video • Fortune Magazine', url: 'https://www.youtube.com/watch?v=AxfmzLz9xXM' }
        ]
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