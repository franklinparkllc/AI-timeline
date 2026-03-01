# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A static, data-driven interactive timeline of AI/ML history (1943–2025). There is **no build toolchain, no npm, no framework** — just vanilla HTML/CSS/JS generated from a Python script that reads a CSV.

## Commands

### Generate the timeline HTML
```bash
python scripts/generate_timeline.py
```
Reads `data/AI_Timeline_1940-2025.csv`, produces `index.html`, and increments `timeline_version.txt`.

### Fix CSV quoting issues (run before regenerating if CSV was edited in Excel)
```bash
python scripts/normalize_csv.py
```

### View locally
```bash
python -m http.server 8000
# Open http://localhost:8000/
```

## Architecture

### Data Flow
```
data/AI_Timeline_1940-2025.csv
    → scripts/generate_timeline.py  (reads, validates, templates)
    → index.html                    (generated static file — do not hand-edit)
    → browser loads styles.css + timeline.js
```

`index.html` is **generated output** — all content changes must go through the CSV and regeneration.

### Frontend (styles.css + timeline.js)

`timeline.js` is the only runtime JS. It handles:
- **Filtering**: search (debounced 300ms), category dropdown, significance level buttons — all read `data-category` and `data-weight` attributes on card elements
- **Navigation**: drag-to-scroll on the timeline wrapper (2× speed), scrubber bar synced to scroll position
- **Parallax**: logo background shifts on scroll

`styles.css` uses CSS custom properties for theming and card layout. Event cards are positioned in alternating top/bottom levels per year column, with weight-based visual hierarchy (weight 3 = landmark = red/bold styling).

### CSV Schema

| Column | Required | Notes |
|--------|----------|-------|
| Year | Yes | Integer |
| Event/Development | Yes | Card body text |
| People/Organizations | Yes | Displayed on card |
| Category | Yes | Must match one of the 8 taxonomy categories exactly |
| Source | Yes | |
| Link | No | URLs with commas must stay quoted — use normalize_csv.py if editing externally |
| Event Weight | No | 1 = minor, 2 = notable, 3 = landmark |

### The 8 Category Taxonomy
`Research Breakthroughs`, `Foundation Models`, `AI Products`, `Generative AI`, `Enterprise & Developer Tools`, `Hardware & Infrastructure`, `Autonomous Systems`, `Governance & Society`

Category names must match exactly — the generator and JS filtering are case-sensitive.

## Key Files

| File | Role |
|------|------|
| `data/AI_Timeline_1940-2025.csv` | **Primary source of truth** — edit this to add/change events |
| `scripts/generate_timeline.py` | HTML generator; contains CSV validation and Jinja-style templating inline |
| `timeline.js` | All interactive behavior |
| `styles.css` | All visual styling |
| `index.html` | Generated output — regenerate rather than editing directly |
