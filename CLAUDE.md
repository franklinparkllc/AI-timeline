# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A static, data-driven interactive timeline of AI/ML history (1943–2026). There is **no build toolchain, no npm, no framework** — just vanilla HTML/CSS/JS generated from a Python script that reads YAML.

## Commands

### Generate the timeline HTML
```bash
python3 scripts/generate_timeline.py
```
Reads `data/timeline.yaml`, produces `index.html`, and increments `timeline_version.txt`.

### View locally
```bash
python3 -m http.server 8000
# Open http://localhost:8000/
```

## Architecture

### Data Flow
```
data/timeline.yaml
    → scripts/generate_timeline.py  (reads, validates, templates)
    → index.html                    (generated static file — do not hand-edit)
    → browser loads styles.css + timeline.js
```

`index.html` is **generated output** — all content changes must go through YAML and regeneration.

### Frontend (styles.css + timeline.js)

`timeline.js` is the only runtime JS. It handles:
- **Filtering**: search (debounced 300ms), category dropdown, significance level buttons — all read `data-category` and `data-weight` attributes on card elements
- **Navigation**: drag-to-scroll on the timeline wrapper (2× speed), scrubber bar synced to scroll position

`styles.css` uses CSS custom properties for theming and card layout. Event cards are positioned in alternating top/bottom levels per year column, with weight-based visual hierarchy (weight 3 = landmark = red/bold styling).

### YAML Schema

| Field | Required | Type | Notes |
|-------|----------|------|-------|
| year | Yes | integer | Range: 1943–2026 |
| event | Yes | string | Card body text; commas/quotes require no escaping |
| people | Yes | string | Semicolon-delimited for multiple people |
| category | Yes | string | Must match one of the 8 taxonomy values exactly (validated) |
| source | Yes | string | `Original` or `Added` — tracks provenance |
| link | No | string | Full HTTPS URL, no escaping needed |
| weight | No | integer | 1 = minor, 2 = notable, 3 = landmark |

### The 8 Valid Categories (enum)
```
Research Breakthroughs
Foundation Models
AI Products
Generative AI
Enterprise & Developer Tools
Hardware & Infrastructure
Autonomous Systems
Governance & Society
```

Category names must match exactly (case-sensitive). The generator validates all categories and skips invalid entries.

## Key Files

| File | Role |
|------|------|
| `data/timeline.yaml` | **Primary source of truth** — edit this to add/change events |
| `scripts/generate_timeline.py` | HTML generator; reads YAML, validates, outputs HTML |
| `timeline.js` | All interactive behavior |
| `styles.css` | All visual styling |
| `index.html` | Generated output — regenerate rather than editing directly |
