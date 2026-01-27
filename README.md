# AI Timeline Interactive Website

An interactive, beautiful HTML timeline visualization of AI and ML developments from 1940-2025.

## Project Structure

```
AI-timeline/
├── data/
│   └── AI_Timeline_1940-2025.csv    # Source CSV data (maintain this)
├── src/
│   ├── index.html                    # Generated HTML timeline
│   ├── styles.css                    # Timeline styling
│   └── script.js                     # Interactive functionality
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
   - Open `src/index.html` in your web browser
   - Or serve it with a local server: `python -m http.server 8000` (then visit http://localhost:8000/src/)

## Data Format

The CSV file should have the following columns:
- **Year**: The year of the event
- **Event/Development**: Description of the event
- **People/Organizations**: Key people or organizations involved
- **Category**: Category of the event (e.g., Neural Networks, NLP, Theory, etc.)
- **Source**: Source of the information
- **New Entry**: Whether this is a new entry (Yes/No)
- **Link**: URL to more information about the event (optional)
- **Event Weight**: Significance/weight of the event for filtering (optional, e.g., "High", "Medium", "Low")

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
