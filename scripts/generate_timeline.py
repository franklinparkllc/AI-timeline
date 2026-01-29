#!/usr/bin/env python3
"""
Generate interactive HTML timeline from CSV data
"""
import csv
import os
import json
from pathlib import Path

def _sanitize_link(link):
    """Ensure link is a single URL; strip any trailing comma + weight that bled in."""
    if not link:
        return ''
    link = link.strip()
    # If link ends with comma + digits (weight bled in), remove it
    if link and link[-1].isdigit():
        i = len(link) - 1
        while i >= 0 and link[i].isdigit():
            i -= 1
        if i >= 0 and link[i] == ',':
            link = link[:i].strip()
    return link if link.startswith(('http://', 'https://')) else link

def _sanitize_weight(weight):
    """Ensure weight is a single number (1-3: minor/major/landmark); drop any URL fragment that bled in."""
    if not weight:
        return ''
    weight = weight.strip()
    # Take only leading digits
    digits = []
    for c in weight:
        if c.isdigit():
            digits.append(c)
        elif digits:
            break
    return ''.join(digits) if digits else ''

def read_csv_data(csv_path):
    """Read timeline data from CSV file"""
    events = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            link = row.get('Link', '').strip()
            weight = row.get('Event Weight', '').strip()
            event = {
                'year': int(row['Year']) if row['Year'].strip() else None,
                'event': row['Event/Development'].strip(),
                'people': row['People/Organizations'].strip(),
                'category': row['Category'].strip(),
                'source': row['Source'].strip(),
                'link': _sanitize_link(link),
                'eventWeight': _sanitize_weight(weight)
            }
            if event['year']:
                events.append(event)
    
    # Sort by year
    events.sort(key=lambda x: x['year'])
    return events

def get_and_increment_version(project_root):
    """Read version from timeline_version.txt, increment, write back; return new version."""
    version_file = project_root / "timeline_version.txt"
    try:
        version = int(version_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        version = 0
    version += 1
    version_file.write_text(str(version) + "\n")
    return version


def generate_html(events, output_path, version=1):
    """Generate HTML timeline from events data"""
    
    # Get unique categories for filtering
    categories = sorted(set(e['category'] for e in events if e['category']))
    
    # Get year range
    years = [e['year'] for e in events if e['year']]
    min_year = min(years) if years else 1940
    max_year = max(years) if years else 2025
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Timeline (1940-2025)</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>AI & ML Timeline</h1>
            <p class="subtitle">Key Developments from {min_year} to {max_year}</p>
        </header>
        
        <div class="controls-container">
            <button class="controls-toggle" id="controlsToggle" title="Filter & Search">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="2" y1="14" x2="6" y2="14"></line><line x1="10" y1="8" x2="14" y2="8"></line><line x1="18" y1="16" x2="22" y2="16"></line></svg>
            </button>
            <div class="controls" id="controlsMenu">
                <div class="search-box">
                    <input type="text" id="searchInput" placeholder="Search events...">
                </div>
                
                <div class="filter-box">
                    <select id="categoryFilter">
                        <option value="">All Categories</option>
"""
    
    for category in categories:
        html += f'                        <option value="{category}">{category}</option>\n'
    
    html += f"""                    </select>
                </div>
                
                <div class="year-navigation">
                    <input type="number" id="yearJump" min="{min_year}" max="{max_year}" placeholder="Year">
                    <button id="jumpBtn">Go</button>
                </div>
            </div>
        </div>

        <div class="parallax-bg">
            <img src="data/FranklinPark_Logo_blue.png" alt="" class="bg-logo" id="parallaxLogo">
        </div>

        <div class="significance-control">
            <label for="significanceSlider" class="significance-label">Significance</label>
            <input type="range" id="significanceSlider" min="1" max="3" value="2" step="1" title="1 = all events, 3 = landmarks only" aria-valuemin="1" aria-valuemax="3" aria-valuenow="2">
            <span class="significance-hint" aria-hidden="true">1 = all · 3 = landmarks</span>
        </div>
        
        <div class="stats">
            <span id="eventCount">{len(events)}</span> events displayed
            <span class="timeline-version" aria-hidden="true"> · v.{version}</span>
        </div>
        
        <div class="timeline-wrapper" id="timelineWrapper">
            <div class="timeline" id="timeline">
"""

    # Group events by year
    events_by_year = {}
    for event in events:
        year = event['year']
        if year not in events_by_year:
            events_by_year[year] = []
        events_by_year[year].append(event)
    
    # Generate timeline entries
    for year in sorted(events_by_year.keys()):
        year_events = events_by_year[year]
        html += f'            <div class="timeline-year" data-year="{year}">\n'
        html += f'                <div class="year-marker">{year}</div>\n'
        html += '                <div class="year-events">\n'
        
        for i, event in enumerate(year_events):
            weight_attr = f' data-weight="{event["eventWeight"]}"' if event['eventWeight'] else ''
            # Assign level (0, 1, 2) to alternate vertical positions
            level = i % 3
            html += f'                    <div class="event-card level-{level}" data-category="{event["category"]}"{weight_attr}>\n'
            html += f'                        <div class="event-content">\n'
            html += f'                            <h3 class="event-title">{event["event"]}</h3>\n'
            
            if event['people']:
                html += f'                            <p class="event-people">{event["people"]}</p>\n'
            
            html += f'                            <div class="event-meta">\n'
            html += f'                                <span class="event-category">{event["category"]}</span>\n'
            html += '                            </div>\n'
            
            if event['link']:
                html += f'                            <div class="event-link">\n'
                html += f'                                <a href="{event["link"]}" target="_blank" rel="noopener noreferrer">Details →</a>\n'
                html += '                            </div>\n'
            
            html += '                        </div>\n'
            html += '                    </div>\n'
        
        html += '                </div>\n'
        html += '            </div>\n'
    
    html += """            </div>
        </div>
        <div class="timeline-scrubber-wrap" aria-label="Timeline position">
            <input type="range" id="timelineScrubber" class="timeline-scrubber" min="0" max="100" value="0" step="0.1" title="Drag to scroll timeline">
        </div>
    </div>
    
    <script src="script.js"></script>
</body>
</html>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Generated HTML timeline with {len(events)} events")
    print(f"Output: {output_path}")

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "AI_Timeline_1940-2025.csv"
    output_path = project_root / "index.html"
    
    # Create directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Please ensure the CSV file exists in the data/ directory")
        return
    
    # Version: increment on each run
    version = get_and_increment_version(project_root)
    print(f"Timeline version: {version}")
    
    # Read and generate
    print(f"Reading CSV from {csv_path}...")
    events = read_csv_data(csv_path)
    print(f"Found {len(events)} events")
    
    print(f"Generating HTML timeline...")
    generate_html(events, output_path, version=version)
    print("Done!")

if __name__ == "__main__":
    main()
