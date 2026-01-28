# AI Timeline Interactive Website

An interactive, beautiful HTML timeline visualization of AI and ML developments from 1940-2025.

## Project Structure

```
AI-timeline/
├── data/
│   └── AI_Timeline_1940-2025.csv    # Source CSV data (maintain this)
├── index.html                    # Generated HTML timeline
├── styles.css                    # Timeline styling
├── script.js                     # Interactive functionality
├── scripts/
│   └── generate_timeline.py         # Script to generate HTML from CSV
└── README.md                         # This file
```

## Quick Start

1. **Update the timeline data:**
   - Edit `data/AI_Timeline_1940-2025.csv` with your updates

2. **Generate the HTML timeline:**
   ```bash
   python scripts/generate_timeline.py
   ```

3. **Open the timeline:**
   - Open `index.html` in your web browser
   - Or serve it with a local server: `python -m http.server 8000` (then visit http://localhost:8000/)

## Category Taxonomy

Events are organized using an **Industry & Impact-focused taxonomy** with 8 key categories:

1. **Research Breakthroughs** - Academic papers, fundamental discoveries, algorithms, theory
2. **Foundation Models** - LLMs, transformers, large-scale pre-training, NLP systems
3. **AI Products** - Consumer apps, assistants, tools, platforms, adoption metrics
4. **Generative AI** - Text-to-image, text-to-video, GANs, diffusion models, multimodal
5. **Enterprise & Developer Tools** - Frameworks, APIs, coding assistants, development platforms
6. **Hardware & Infrastructure** - GPUs, TPUs, custom chips, compute clusters
7. **Autonomous Systems** - Self-driving cars, robots, agents, game-playing AI
8. **Governance & Society** - Regulation, ethics, safety research, business developments

This taxonomy provides a clear narrative for understanding AI's evolution from research to real-world impact.

## Data Format

The CSV file should have the following columns:
- **Year**: The year of the event
- **Event/Development**: Description of the event
- **People/Organizations**: Key people or organizations involved
- **Category**: Category of the event (must be one of the 8 taxonomy categories above)
- **Source**: Source of the information
- **Link**: URL to more information about the event (optional)
- **Event Weight**: Significance/weight of the event for filtering (optional, e.g., "High", "Medium", "Low")

## Audit the Timeline (copy/paste prompt)

Use this prompt with your preferred AI assistant to review and improve a slice of the CSV timeline.

```text
Take a look at @data/AI_Timeline_1940-2025.csv. It serves a script that builds an interactive AI timeline.

Please audit entries from 2025 through today:
1) Flag entries that are poorly labeled or mis-entered (wrong year, wrong people/org, wrong category).
2) Add key events/papers/research that are missing in this timeframe (include credible links and focus on primary sources).
3) If an item is irrelevant/insignificant for the timeline narrative, set its Event Weight to 0.

Constraints:
- Keep the existing CSV columns and formatting.
- Use numeric Event Weight on a 0–10 scale (0 = irrelevant).
- Prefer “Person + org/lab” in People/Organizations when known.
Then regenerate the timeline with: python scripts/generate_timeline.py
```

## Features

- **Interactive Timeline**: Scroll through years with smooth navigation
- **Filtering**: Filter events by category and event weight
- **Search**: Search for specific events, people, or organizations
- **Links**: Click "Learn More" links to get additional information about events
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Beautiful UI**: Modern, clean design with smooth animations
- **Year Navigation**: Jump to specific years quickly

## Maintenance

### Adding New Events

1. Open `data/AI_Timeline_1940-2025.csv` in a text editor or spreadsheet application
2. Add a new row with the event details
3. Run `python scripts/generate_timeline.py` to regenerate the HTML
4. Refresh your browser to see the updates

## Requirements

- Python 3.6+

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## License

This project is for research and educational purposes.
