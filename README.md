# AI Timeline Interactive Website

An interactive, beautiful HTML timeline visualization of AI and ML developments from 1943–2026.

## Project Structure

```
AI-timeline/
├── data/
│   └── timeline.yaml                # Source YAML data (edit this)
├── index.html                       # Generated HTML timeline
├── styles.css                       # Timeline styling
├── timeline.js                      # Interactive functionality
├── scripts/
│   └── generate_timeline.py         # Script to generate HTML from YAML
└── README.md                        # This file
```

## Quick Start

1. **Update the timeline data:**
   - Edit `data/timeline.yaml` with your updates (YAML format, no quoting issues)

2. **Generate the HTML timeline:**
   ```bash
   python3 scripts/generate_timeline.py
   ```

3. **Open the timeline:**
   - Open `index.html` in your web browser
   - Or serve it with a local server: `python3 -m http.server 8000` (then visit http://localhost:8000/)

## Data Format

The YAML file should be a list of events, each with the following fields:

```yaml
- year: 1943
  event: "A Logical Calculus of the Ideas Immanent in Nervous Activity"
  people: "Warren McCulloch; Walter Pitts (University of Chicago)"
  category: Research Breakthroughs
  source: Original
  link: https://www.cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf
  weight: 3
```

| Field | Required | Type | Notes |
|-------|----------|------|-------|
| year | Yes | integer | Event year (1943–2026) |
| event | Yes | string | Event description (no escaping needed) |
| people | Yes | string | Key people/orgs; semicolon-delimited for multiple |
| category | Yes | string | Must match one of the 8 taxonomy categories (validated) |
| source | Yes | string | `Original` or `Added` — marks data provenance |
| link | No | string | Full HTTPS URL to more information |
| weight | No | integer | Event significance: 1 (minor), 2 (notable), 3 (landmark) |

**YAML Advantages:** No escaping needed for commas or quotes in field values. Edit safely without worrying about CSV formatting issues.

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

## Features

- **Interactive Timeline**: Scroll through years with smooth horizontal navigation
- **Significance Filter**: Filter events by impact (All → Notable → Landmarks)
- **Filtering**: Filter events by category and event weight
- **Search**: Search for specific events, people, or organizations
- **Links**: Direct links to primary sources (arXiv, Nature, official blogs) for major milestones
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Beautiful UI**: Modern, clean design with smooth animations and a gradient-styled control widget
- **Year Navigation**: Jump to specific years with the interactive scrubber bar

## Maintenance

### Adding or Editing Events

1. Open `data/timeline.yaml` in any text editor
2. Add or modify event entries in YAML format
3. Run `python3 scripts/generate_timeline.py` to regenerate the HTML
4. Refresh your browser to see the updates

**Note:** YAML is safe from CSV quoting issues. Commas, quotes, and special characters in field values require zero escaping.

## Requirements

- Python 3.6+
- PyYAML: `pip install pyyaml`

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## License

This project is for research and educational purposes.
