// Model Basics - Card Data
// All content for the topic cards
// Incorporating "Factory/Skyscraper" and "Flavor Profile" analogies

const cardsData = [
    {
        category: 'arch',
        badge: 'Overview',
        title: '1. Introduction to modern AI models',
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
        title: '2. Before Transformers',
        description: 'Understanding why the Transformer architecture was revolutionary requires looking at what came before.',
        paragraphs: [
            'Early neural networks for language processing used <strong>Recurrent Neural Networks (RNNs)</strong> and <strong>Long Short-Term Memory (LSTM)</strong> architectures. These models processed text sequentially, one word at a time, maintaining context through a resource-intensive process.',
            'The problem? Sequential processing was slow and could not parallelize across GPUs. Models also struggled with long-range dependencies. By the time an RNN reached word 50, it had largely forgotten word 1.',
            '<strong>Seq2Seq</strong> (encoder-decoder) models added <strong>attention</strong> mechanisms, first for machine translation, so the decoder could look at relevant parts of the input. Attention helped, but the backbone was still sequential RNNs. These precursors started appearing around 2014, with the first successful implementation of attention in 2015 by Bahdanau et al., followed by the breakthrough Vaswani et al. paper in 2017.',
            'The 2017 paper <em>Attention Is All You Need</em> introduced the <strong>Transformer</strong>, which dropped the RNN entirely. It processes all tokens in parallel using only attention mechanisms, enabling the massive scale we see in modern AI.'
        ],
        bullets: [
            '<strong>RNNs and LSTMs:</strong> Sequential processing (slow), vanishing gradients (poor long-term memory)',
            '<strong>Seq2Seq:</strong> Encoder-decoder plus attention (for tasks like translation), but RNNs remained the bottleneck',
            '<strong>The Breakthrough:</strong> Transformers keep attention, drop recurrence. Every token can look at every other token in parallel'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Why This Matters:</strong> The shift from sequential to parallel processing is why modern AI could scale to billions of parameters and trillion-token datasets. RNNs could not scale effectively. Transformers could.'
        },
        resources: [
            { icon: '📺', title: 'Transformers and Attention Overview', meta: '58 min • Deep dive into transformers', url: 'https://www.youtube.com/watch?v=KJtZARuO3JY' },
            { icon: '🌐', title: 'Attention? Attention!', meta: 'Lilian Weng', url: 'https://lilianweng.github.io/posts/2018-06-24-attention/' }        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '3. Tokens & Tokenization',
        description: 'Models don\'t read text directly—they process numeric token IDs that represent pieces of words.',
        paragraphs: [
            'AI models operate on numbers, not letters. A <strong>tokenizer</strong> converts text into integer IDs representing vocabulary fragments. For example, "Ingenious" might split into three tokens: <code>In</code>, <code>gen</code>, and <code>ious</code>.',
            'This approach (often <strong>BPE</strong>—<strong>Byte Pair Encoding</strong>—or similar) balances efficiency and flexibility: common words stay whole, rare words split into reusable parts.',
            'BPE works at the byte level, using the 256 possible byte values (0-255) as a universal foundation. This allows it to represent any language, emoji, or Unicode character through UTF-8 encoding, making it language-agnostic.',
            'Multimodal models do the same idea for images (patches) and audio (chunks).'
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
            { icon: '📺', title: 'Byte Pair Encoding Tokenization', meta: '7 min • Tokenization & BPE explained', url: 'https://www.youtube.com/watch?v=4A_nfXyBD08' },
            { icon: '📺', title: 'Build GPT Tokenizer', meta: '2h 13min • Andrej Karpathy', url: 'https://www.youtube.com/watch?v=zduSFxRajkE' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '4. From Numbers to Meaning: Embeddings',
        description: 'Token IDs are just numbers—embeddings transform them into rich, meaningful representations the model can process.',
        paragraphs: [
            'After tokenization, the model has a sequence of token IDs—integers like [4829, 2121, 8945]. But numbers alone are meaningless. The model needs to understand what each token <em>represents</em>.',
            'Enter <strong>embeddings</strong>: the model looks up each token ID in a massive learned table (the <strong>embedding matrix</strong>) and retrieves its corresponding <strong>vector</strong>—a list of hundreds or thousands of numbers. Each dimension captures some aspect of meaning: semantic properties, grammatical role, contextual patterns learned during training.',
            'These vectors live in <strong>embedding space</strong>, a high-dimensional coordinate system where similar meanings cluster together. "Dog" and "puppy" sit close to each other. "King" and "queen" differ primarily along a "gender" axis. This geometry is how the model "understands" relationships without explicit rules.',
            'The model also adds <strong>positional encodings</strong>—patterns that tell it where each token appears in the sequence. Without position, "dog bites man" and "man bites dog" would look identical. Position + meaning = the full input representation that flows into the Transformer layers.'
        ],
        bullets: [
            '<strong>Embedding Lookup:</strong> Token ID → retrieve learned vector from embedding table',
            '<strong>High-Dimensional Space:</strong> Vectors typically have 768, 1024, 4096+ dimensions',
            '<strong>Learned During Training:</strong> The embedding table is optimized alongside the rest of the model',
            '<strong>Positional Encoding:</strong> Adds order information so word sequence matters'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>Analogy:</strong> If tokenization assigns each word a locker number, embeddings are the contents of that locker—a profile describing the word\'s meaning, usage, and relationships. Position encoding adds a timestamp: when that locker was opened in the sequence.'
        },
        resources: [
            { icon: '📺', title: 'Language Models: Tokens and Embeddings', meta: '7 min • Visual explanation', url: 'https://www.youtube.com/watch?v=izbifbq3-eI' },
            { icon: '📺', title: 'Language Models & Transformers', meta: '20 min • Computerphile', url: 'https://www.youtube.com/watch?v=rURRYI66E54' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '5. The Transformer Architecture',
        description: 'The Transformer is a stack of repeating layers that progressively refine token representations through attention and processing.',
        paragraphs: [
            'A Transformer isn\'t a single operation—it\'s a <strong>stack of identical layers</strong> (typically 12, 24, 48, or more) that process embeddings sequentially. Think of it as a skyscraper: each floor performs the same two operations on every token\'s vector as it passes through.',
            '<strong>Each Transformer layer contains:</strong> (1) A <strong>self-attention mechanism</strong> that lets tokens "communicate" and update their representations based on context, and (2) A <strong>feed-forward network</strong> that processes each token\'s vector independently, refining its meaning.',
            'As vectors flow upward through the stack, they accumulate increasingly abstract and context-aware information. Early layers capture basic patterns like grammar and syntax. Middle layers learn relationships and simple logic. Deep layers encode complex reasoning, nuanced meaning, and task-specific behavior.',
            'The power of Transformers comes from this <strong>deep, repeated processing</strong>. Each layer adds a small refinement, but stacking dozens of them allows the model to build sophisticated representations from simple token embeddings.'
        ],
        bullets: [
            '<strong>Layer Structure:</strong> Self-attention + feed-forward network, repeated N times',
            '<strong>Progressive Refinement:</strong> Each layer adds context and abstraction to token vectors',
            '<strong>Residual Connections:</strong> Original information is preserved as it flows upward, preventing information loss',
            '<strong>Depth = Capability:</strong> More layers enable more complex reasoning and pattern recognition'
        ],
        callout: {
            type: 'insight',
            content: '<strong>The Skyscraper Analogy:</strong> Ground floor tokens know only their own meaning. As they ride the elevator through dozens of floors—each adding context from surrounding words—they emerge at the top with rich, nuanced understanding of their role in the specific sentence.'
        },
        resources: [
            { icon: '🌐', title: 'The Illustrated Transformer', meta: 'Jay Alammar • Visual explanation', url: 'https://jalammar.github.io/illustrated-transformer/' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '6. Inside a Transformer Layer',
        description: 'Each layer combines attention, processing, and stabilization techniques to refine token representations without losing information.',
        paragraphs: [
            'We said each Transformer layer has two main components—but the full picture includes critical "glue" that makes deep stacking possible. Here\'s what actually happens as tokens flow through a single layer:',
            '<strong>1. Layer Normalization (Pre-norm):</strong> Before processing, token vectors are normalized to have consistent scale. This stabilizes training in deep networks and prevents values from exploding or vanishing as they move through dozens of layers.',
            '<strong>2. Self-Attention:</strong> Tokens "talk" to each other (covered in detail in the next slide). Each token\'s vector is updated based on the entire sequence context.',
            '<strong>3. Residual Connection (Add):</strong> The original input vector is added back to the attention output. This "skip connection" preserves information from earlier layers and enables training of very deep networks (up to 100+ layers).',
            '<strong>4. Layer Normalization (again):</strong> Normalize again before the next component.',
            '<strong>5. Feed-Forward Network:</strong> Each token\'s vector passes through a small neural network (typically: expand to 4x size, apply non-linearity, compress back). This processes each token independently, refining its representation.',
            '<strong>6. Residual Connection (again):</strong> Add the original vector again before passing to the next layer.',
            'This pattern—normalize, process, add residual—appears twice per layer and is what allows Transformers to scale to enormous depths without training instability.'
        ],
        bullets: [
            '<strong>Feed-Forward Network:</strong> Expands each vector to 4x size, transforms it, compresses back—adding non-linear processing power',
            '<strong>Residual Connections:</strong> Like highway lanes that bypass each floor, ensuring original info always flows upward',
            '<strong>Layer Normalization:</strong> Keeps vector values in a healthy range, prevents training collapse in deep models',
            '<strong>Why It Matters:</strong> Without residuals and normalization, training 48-layer models would be nearly impossible'
        ],
        callout: {
            type: 'note',
            content: '<strong>The Highway Analogy:</strong> Attention and feed-forward networks transform vectors (like taking side streets), but residual connections provide an express highway for the original information. This prevents the model from "forgetting" earlier insights as tokens travel through dozens of layers.'
        },
        resources: [
            { icon: '📺', title: 'Layer Normalization Explained', meta: '8 min • Visual explanation', url: 'https://www.youtube.com/watch?v=2V3Ud-FnvUs' },
            { icon: '🌐', title: 'The Illustrated Transformer', meta: 'Jay Alammar • Visual explanation', url: 'https://jalammar.github.io/illustrated-transformer/' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '7. Attention: How Tokens Communicate',
        description: 'Self-attention is the key mechanism that lets each token update its representation by looking at all other tokens in the sequence.',
        paragraphs: [
            'Attention is THE innovation that makes Transformers work. It solves the problem RNNs couldn\'t: how can every token simultaneously look at every other token to understand context?',
            '<strong>The Mechanism—Query, Key, Value (Q, K, V):</strong> For each token, the model creates three vectors: a <strong>Query</strong> (what am I looking for?), a <strong>Key</strong> (what do I represent?), and a <strong>Value</strong> (what information do I carry?). Each token\'s Query is compared to every other token\'s Key to calculate <strong>attention scores</strong>—how much should I pay attention to each other word?',
            'These scores are normalized (via softmax) into weights that sum to 1.0. Then, the model computes a weighted average of all Value vectors, using these attention scores as weights. The result? Each token\'s vector is updated to incorporate relevant information from the entire sequence.',
            '<strong>Example:</strong> In "The bank by the river was flooded," the token "bank" compares its Query to the Keys of all other tokens. "River" and "flooded" produce high attention scores (Keys match the Query), so "bank" pulls in their Value vectors. The generic "bank" embedding morphs into "financial institution" or "riverbank" based on context.',
            '<strong>Multi-Head Attention:</strong> Models don\'t run attention once—they run it multiple times in parallel (8, 16, or 32 "heads"). Each head learns to focus on different relationships: syntax, semantics, coreference, etc. Outputs are concatenated and mixed, giving the model multiple simultaneous "perspectives" on the sequence.',
            '<strong>Why It\'s Called Self-Attention:</strong> Tokens attend to other tokens in the same sequence—the input attends to itself. (Later models also use cross-attention between different sequences, like encoder-decoder architectures.)'
        ],
        bullets: [
            '<strong>Query, Key, Value:</strong> Each token generates three vectors via learned transformations',
            '<strong>Attention Scores:</strong> Dot product of Query with all Keys, normalized by softmax',
            '<strong>Weighted Sum:</strong> Combine all Value vectors using attention scores as weights',
            '<strong>Multi-Head:</strong> Run attention multiple times in parallel to capture different relationships',
            '<strong>Parallel Processing:</strong> All tokens compute attention simultaneously—no sequential bottleneck like RNNs'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>The Cocktail Party:</strong> Imagine a room where everyone shouts their interests (Keys). You shout what you\'re looking for (Query), and the room\'s acoustics amplify voices that match. You hear a weighted mix of conversations (Values), with relevant voices louder. Multi-head attention is like having multiple conversations simultaneously—one about work, one about hobbies, one about mutual friends.'
        },
        resources: [
            { icon: '📺', title: '3Blue1Brown: Attention in Transformers', meta: '26 min • Animated explanation', url: 'https://www.youtube.com/watch?v=eMlx5fFNoYc' },
            { icon: '🎬', title: 'Attention Is All You Need', meta: '15 min • Visual walkthrough', url: 'https://www.youtube.com/watch?v=wjZofJX0v4M' },
            { icon: '🌐', title: 'The Illustrated Transformer', meta: 'Jay Alammar • Attention visualizations', url: 'https://jalammar.github.io/illustrated-transformer/' }
        ]
    },
    {
        category: 'train',
        badge: 'Training',
        title: '8. Pre-Training',
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
        title: '9. Post-Training',
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
        title: '10. Bias, Fairness & Limitations',
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
            { icon: 'ðŸ“º', title: 'AI Bias Explained', meta: '9 min â€¢ TEDx', url: 'https://www.youtube.com/watch?v=59bMh59JQDo' }
        ]
    },
    {
        category: 'infer',
        badge: 'Inference',
        title: '11. The Frozen State',
        description: 'After training, model weights are frozenâ€”inference runs data through this fixed architecture without learning.',
        paragraphs: [
            'Once training completes, the model\'s weights are <strong>locked</strong>. Inference (generating responses) reads these weights but never modifies them. This is why chatting doesn\'t teach the model anything permanentâ€”corrections only affect the current conversation\'s context.',
            '<strong>In-Context Learning</strong> is the workaround: if you provide examples in your prompt, the model adapts its outputs to match those patternsâ€”but only for that single interaction. After the conversation ends, the model "forgets" everything.'
        ],
        bullets: [
            'No learning during inference: feedback doesn\'t update weights',
            'Context window is the only "memory"â€”once it\'s full, earlier messages get truncated',
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
        title: '12. The Prompt Stack',
        description: 'Every model interaction involves a carefully structured stack of hidden instructions, history, and user input.',
        paragraphs: [
            'When you send a message, the model doesn\'t just see your text. It processes a <strong>prompt stack</strong> with multiple layers:',
            '<strong>1. System Prompt:</strong> Hidden instructions defining the model\'s persona and behavior ("You are a helpful AI assistantâ€¦").',
            '<strong>2. Conversation History:</strong> Every prior message in the conversation is re-sent on each turn, consuming context window space.',
            '<strong>3. User Prompt:</strong> Your actual message, often wrapped in XML tags for parsing (<code>&lt;user_query&gt;</code>).'
        ],
        bullets: [
            'Long conversations get slower and more expensiveâ€”the entire history is typically reprocessed every turn',
            'At context limit, early messages are dropped (the model "forgets" the start of long chats)',
            '<strong>Context rot:</strong> In long chats, constraints can get buried or droppedâ€”leading to drift and contradictions',
            'Prompt engineering exploits this structure: clear instructions, examples, and formatting improve outputs'
        ],
        callout: {
            type: 'note',
            content: '<strong>Note:</strong> This is why long conversations become expensive and slow. Each new response requires reprocessing thousands of prior tokens. <strong>Tip:</strong> Periodically restate the goal and key constraints (or start a fresh thread with a short summary) to reduce drift.'
        },
        resources: [
            { icon: 'ðŸ“˜', title: 'Claude prompting best practices', meta: 'Anthropic docs â€¢ Prompt engineering', url: 'https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices' }
        ]
    },
    {
        category: 'infer',
        badge: 'Inference',
        title: '13. The Selection Dice Roll',
        description: 'The final step: turning a "massaged" vector back into a human word.',
        paragraphs: [
            'At the roof of the skyscraper, the model has a highly refined vector. It compares this "thought" against its entire vocabulary and gives every word a score (<strong>Logits</strong>).',
            'These scores are turned into probabilities. The model doesn\'t "know" the answer; it just knows that "Medici" has a 75% chance of being the next right word.'
        ],
        bullets: [
            '<strong>Temperature:</strong> Controls the "risk." High temperature = roll the dice on lower-probability words (creativity)',
            '<strong>Autoregressive:</strong> The model picks one token, adds it to the prompt, and runs the entire skyscraper again for the next one',
            '<strong>Streaming:</strong> Why you see text appear word-by-word'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Prediction, not Truth:</strong> The model is optimized for "plausibility." If the most statistically likely next word is a hallucination, the model will pick it because its math told it to, not because it "wants" to lie.'
        },
        resources: [
            { icon: 'ðŸ“º', title: 'Why Large Language Models Hallucinate', meta: 'Video â€¢ Practical explanation', url: 'https://www.youtube.com/watch?v=cfqtFvWOfg0' },
            { icon: 'ðŸ§ ', title: 'Why language models hallucinate', meta: 'OpenAI â€¢ Research explainer', url: 'https://openai.com/index/why-language-models-hallucinate/' },
            { icon: 'ðŸ“„', title: 'Mata v. Avianca (court filing with fabricated citations)', meta: 'Primary source â€¢ SDNY docket', url: 'https://law.justia.com/cases/federal/district-courts/new-york/nysdce/1:2022cv01461/575368/54/' },
            { icon: 'ðŸ“°', title: 'Google Bard demo error (JWST claim)', meta: 'Reuters â€¢ Feb 2023', url: 'https://www.reuters.com/technology/google-ai-chatbot-bard-offers-inaccurate-information-company-ad-2023-02-08/' }
        ]
    },
    {
        category: 'infer',
        badge: 'Summary',
        title: '14. What happens when you send a message?',
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
        title: '15. Embedding Models: Semantic Search & Retrieval',
        description: 'Standalone embedding models power semantic search, document retrieval, and RAG systems by measuring meaning similarity.',
        paragraphs: [
            'While LLMs use embeddings internally (as we saw in the architecture section), <strong>embedding models</strong> are specialized tools trained specifically to convert text into vectors optimized for similarity comparison. Unlike generative models, they don\'t produce text—they produce numerical representations designed for search and matching.',
            '<strong>How they work:</strong> You feed text (a query, document, or sentence) into an embedding model, and it outputs a fixed-size vector (typically 384, 768, or 1536 dimensions). Documents with similar meaning produce similar vectors—measured by <strong>cosine similarity</strong> or dot product. "How do I reset my password?" and "Password reset instructions" score highly similar, even with no word overlap.',
            '<strong>Semantic Search:</strong> Convert all documents in a database to embeddings (done once). When a user queries, convert their question to an embedding and find the closest document vectors. This finds relevant results by <em>meaning</em>, not keyword matching. A search for "infant medicine dosage" will surface documents about "pediatric pharmaceutical guidelines."',
            '<strong>Vector Databases:</strong> Specialized databases (Pinecone, Weaviate, Chroma, FAISS) store embeddings and enable fast similarity search across millions of vectors. They use approximate nearest neighbor (ANN) algorithms to find close matches quickly, making real-time semantic search practical at scale.',
            '<strong>Critical for RAG:</strong> Retrieval-Augmented Generation systems use embedding models to find relevant documents, then pass them to an LLM for generation. The embedding model handles the "find" step; the LLM handles the "generate" step.'
        ],
        bullets: [
            '<strong>Dedicated Models:</strong> OpenAI text-embedding-3, Cohere embed-v3, sentence-transformers—optimized for similarity, not generation',
            '<strong>Applications:</strong> Document search, recommendation systems, duplicate detection, clustering, question-answering retrieval',
            '<strong>Multimodal Embeddings:</strong> CLIP-style models map images and text to the same space—search images with text queries',
            '<strong>Fine-tuning:</strong> Embedding models can be specialized for domain-specific search (legal docs, medical records, code)'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Why Not Just Use an LLM?</strong> Embedding models are 100x faster and cheaper than running an LLM for every document. They\'re designed for one task—measuring similarity—and do it extremely efficiently. For RAG systems, you embed documents once, then reuse those embeddings for millions of queries.'
        },
        resources: [
            { icon: '🎬', title: 'Embeddings Explained', meta: '18 min • 3D visualizations', url: 'https://www.youtube.com/watch?v=eUbKYEC0D3Y' },
            { icon: '🌐', title: 'OpenAI Embeddings Guide', meta: 'Technical documentation', url: 'https://platform.openai.com/docs/guides/embeddings' }
        ]
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: '16. Multimodal Models',
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
        title: '17. RAG (Retrieval-Augmented Generation)',
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
        title: '18. Tool Use (Function Calling)',
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
        title: '19. Reasoning: Two Paradigms',
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
            content: '<strong>The Shift:</strong> Traditional CoT is a user-side prompting trick. Modern reasoning-focused models embed deliberate thinking into the systemâ€”spending extra compute during inference to reduce errors without additional training.'
        },
        resources: [
            { icon: 'ðŸŽ¬', title: 'Chain-of-Thought Explained', meta: '8 min', url: 'https://www.youtube.com/watch?v=AFE6x81AP4k' },
            { icon: 'ðŸ“º', title: 'Test-Time Scaling', meta: '12 min â€¢ Reasoning at inference time', url: 'https://www.youtube.com/watch?v=NbE8MoR8mPw' }
        ]
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: '20. Agentic Workflows',
        description: 'Combining reasoning, tools, and planning creates autonomous agents that can accomplish complex multi-step tasks.',
        paragraphs: [
            'An <strong>agent</strong> is an AI system that can perceive, reason, plan, and act autonomously. By combining inference-time reasoning with tool use, agents break down complex goals into actionable steps.',
            '<strong>ReAct Pattern:</strong> Thought â†’ Action â†’ Observation â†’ Next Thought. The model iteratively reasons, uses tools, observes results, and adjusts its plan.',
            'Examples: Booking a flight (search â†’ check calendar â†’ compare prices â†’ confirm), debugging code (run â†’ read error â†’ fix â†’ re-run), conducting research (search â†’ summarize â†’ synthesize).'
        ],
        bullets: [
            '<strong>Planning:</strong> Decompose "book a trip" into searchable sub-tasks',
            '<strong>Reflection:</strong> Verify outputs ("Does this code compile?" â†’ execute â†’ fix â†’ retry)',
            '<strong>Memory:</strong> Maintain state across multiple interactions (session history, external databases)'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>Analogy:</strong> A standard LLM is a smart person. An agent is that person with a computer, calculator, notepad, and the ability to search the internetâ€”empowered to take action, not just think.'
        }
    },
    {
        category: 'infer',
        badge: 'Conclusion',
        title: '21. Understanding the System',
        description: 'AI models are not consciousâ€”they\'re sophisticated statistical systems that mirror human knowledge.',
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
            content: '<strong>Final Thought:</strong> The "magic" of AI isn\'t that it thinksâ€”it\'s that billions of mathematical operations, trained on trillions of tokens, compress human knowledge into a reusable, frozen artifact. Understanding this transforms you from a passive user into an informed practitioner.'
        }
    }
];
