#!/usr/bin/env python3
"""
Generate interactive HTML timeline from CSV data
"""
import csv
import os
import json
from pathlib import Path

def read_csv_data(csv_path):
    """Read timeline data from CSV file"""
    events = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean up the data
            event = {
                'year': int(row['Year']) if row['Year'].strip() else None,
                'event': row['Event/Development'].strip(),
                'people': row['People/Organizations'].strip(),
                'category': row['Category'].strip(),
                'source': row['Source'].strip(),
                'newEntry': row.get('New Entry', '').strip().lower() == 'yes',
                'link': row.get('Link', '').strip(),
                'eventWeight': row.get('Event Weight', '').strip()
            }
            if event['year']:
                events.append(event)
    
    # Sort by year
    events.sort(key=lambda x: x['year'])
    return events

def generate_html(events, output_path):
    """Generate HTML timeline from events data"""
    
    # Get unique categories for filtering
    categories = sorted(set(e['category'] for e in events if e['category']))
    
    # Get unique event weights for filtering
    event_weights = sorted(set(e['eventWeight'] for e in events if e['eventWeight']), reverse=True)
    
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
        
        <div class="controls">
            <div class="search-box">
                <input type="text" id="searchInput" placeholder="Search events, people, or organizations...">
            </div>
            
            <div class="filter-box">
                <label for="categoryFilter">Filter by Category:</label>
                <select id="categoryFilter">
                    <option value="">All Categories</option>
"""
    
    for category in categories:
        html += f'                    <option value="{category}">{category}</option>\n'
    
    html += """                </select>
            </div>
            
            <div class="filter-box">
                <label for="weightFilter">Filter by Event Weight:</label>
                <select id="weightFilter">
                    <option value="">All Weights</option>
"""
    
    for weight in event_weights:
        html += f'                    <option value="{weight}">{weight}</option>\n'
    
    html += """                </select>
            </div>
            
            <div class="year-navigation">
                <label for="yearJump">Jump to Year:</label>
                <input type="number" id="yearJump" min="1940" max="2025" placeholder="Year">
                <button id="jumpBtn">Go</button>
            </div>
        </div>
        
        <div class="stats">
            <span id="eventCount">{len(events)}</span> events displayed
        </div>
        
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
        
        for event in year_events:
            new_class = ' new-entry' if event['newEntry'] else ''
            weight_attr = f' data-weight="{event["eventWeight"]}"' if event['eventWeight'] else ''
            html += f'                    <div class="event-card{new_class}" data-category="{event["category"]}"{weight_attr}>\n'
            html += f'                        <div class="event-content">\n'
            html += f'                            <h3 class="event-title">{event["event"]}</h3>\n'
            
            if event['people']:
                html += f'                            <p class="event-people"><strong>People/Organizations:</strong> {event["people"]}</p>\n'
            
            html += f'                            <div class="event-meta">\n'
            html += f'                                <span class="event-category">{event["category"]}</span>\n'
            if event['source']:
                html += f'                                <span class="event-source">{event["source"]}</span>\n'
            if event['eventWeight']:
                html += f'                                <span class="event-weight">Weight: {event["eventWeight"]}</span>\n'
            html += '                            </div>\n'
            
            if event['link']:
                html += f'                            <div class="event-link">\n'
                html += f'                                <a href="{event["link"]}" target="_blank" rel="noopener noreferrer">Learn More →</a>\n'
                html += '                            </div>\n'
            
            html += '                        </div>\n'
            html += '                    </div>\n'
        
        html += '                </div>\n'
        html += '            </div>\n'
    
    html += """        </div>
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
    output_path = project_root / "src" / "index.html"
    
    # Create directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Please ensure the CSV file exists in the data/ directory")
        return
    
    # Read and generate
    print(f"Reading CSV from {csv_path}...")
    events = read_csv_data(csv_path)
    print(f"Found {len(events)} events")
    
    print(f"Generating HTML timeline...")
    generate_html(events, output_path)
    print("Done!")

if __name__ == "__main__":
    main()
