# AI Timeline Interactive Website

An interactive, beautiful HTML timeline visualization of AI and ML developments from 1940-2025.

## Project Structure

```
AI-timeline/
├── data/
│   └── AI_Timeline_1940-2025.csv    # Source CSV data (maintain this)
├── index.html                    # Generated HTML timeline
├── styles.css                    # Timeline styling
├── timeline.js                   # Interactive functionality
├── scripts/
│   ├── generate_timeline.py         # Script to generate HTML from CSV
│   └── normalize_csv.py             # Re-quote CSV columns (prevents link/weight bleed)
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

## Features

- **Interactive Timeline**: Scroll through years with smooth horizontal navigation and parallax background.
- **Significance Filter**: A minimal, stylized slider to filter events by impact (All to Landmarks).
- **Branded Parallax**: Subtle background logo that shifts as you navigate the timeline.
- **Filtering**: Filter events by category and event weight.
- **Search**: Search for specific events, people, or organizations.
- **Links**: Direct links to primary sources (arXiv, Nature, official blogs) for major milestones.
- **Responsive Design**: Works on desktop, tablet, and mobile devices.
- **Beautiful UI**: Modern, clean design with smooth animations and a gradient-styled control widget.
- **Year Navigation**: Jump to specific years quickly.

## Maintenance

### Adding New Events

1. Open `data/AI_Timeline_1940-2025.csv` in a text editor or spreadsheet application
2. Add a new row with the event details
3. Run `python scripts/generate_timeline.py` to regenerate the HTML
4. Refresh your browser to see the updates

### Fixing Link / Weight Bleed (CSV quoting)

The CSV stores **Link** and **Event Weight** in quoted columns so URLs that contain commas don’t spill into the weight column. If you edit the CSV in Excel or another tool that strips quotes:

1. Run `python scripts/normalize_csv.py` to re-quote all columns.
2. Then run `python scripts/generate_timeline.py` as usual.

The generator also sanitizes link and weight values when reading, so any existing bleed is corrected when the HTML is built.

## Requirements

- Python 3.6+

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## License

This project is for research and educational purposes.
