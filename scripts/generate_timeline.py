#!/usr/bin/env python3
"""
Generate interactive HTML timeline from CSV data
"""
import csv
import os
import json
from pathlib import Path
from typing import List, Dict, Optional

def _sanitize_link(link: str) -> str:
    """
    Ensure link is a single URL; strip any trailing comma + weight that bled in.
    
    Args:
        link: Link string that may contain trailing weight data
        
    Returns:
        Cleaned link string
    """
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

def _sanitize_weight(weight: str) -> str:
    """
    Ensure weight is a single number (1-3: minor/major/landmark); drop any URL fragment that bled in.
    
    Args:
        weight: Weight string that may contain URL fragments
        
    Returns:
        Cleaned weight string (digits only)
    """
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

def read_csv_data(csv_path: Path) -> List[Dict[str, str]]:
    """
    Read timeline data from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of event dictionaries
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV structure is invalid
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    events = []
    required_columns = ['Year', 'Event/Development', 'People/Organizations', 'Category', 'Source']
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Validate required columns exist
            if not reader.fieldnames:
                raise ValueError("CSV file appears to be empty or invalid")
            
            missing_columns = [col for col in required_columns if col not in reader.fieldnames]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                try:
                    link = row.get('Link', '').strip()
                    weight = row.get('Event Weight', '').strip()
                    
                    # Validate year
                    year_str = row.get('Year', '').strip()
                    if not year_str:
                        print(f"Warning: Row {row_num} has no year, skipping")
                        continue
                    
                    try:
                        year = int(year_str)
                    except ValueError:
                        print(f"Warning: Row {row_num} has invalid year '{year_str}', skipping")
                        continue
                    
                    event = {
                        'year': year,
                        'event': row.get('Event/Development', '').strip(),
                        'people': row.get('People/Organizations', '').strip(),
                        'category': row.get('Category', '').strip(),
                        'source': row.get('Source', '').strip(),
                        'link': _sanitize_link(link),
                        'eventWeight': _sanitize_weight(weight)
                    }
                    
                    # Validate required fields
                    if not event['event']:
                        print(f"Warning: Row {row_num} has no event description, skipping")
                        continue
                    
                    events.append(event)
                except Exception as e:
                    print(f"Warning: Error processing row {row_num}: {e}")
                    continue
    except UnicodeDecodeError as e:
        raise ValueError(f"CSV file encoding error: {e}. Please ensure the file is UTF-8 encoded.")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    if not events:
        raise ValueError("No valid events found in CSV file")
    
    # Sort by year
    events.sort(key=lambda x: x['year'])
    return events

def get_and_increment_version(project_root: Path) -> int:
    """
    Read version from timeline_version.txt, increment, write back; return new version.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        New version number
    """
    version_file = project_root / "timeline_version.txt"
    try:
        version = int(version_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        version = 0
    version += 1
    version_file.write_text(str(version) + "\n")
    return version


def generate_html(events: List[Dict[str, str]], output_path: Path, version: int = 1) -> None:
    """
    Generate HTML timeline from events data.
    
    Args:
        events: List of event dictionaries
        output_path: Path where HTML file should be written
        version: Version number for the timeline
    """
    
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
        
        <div class="toolbar" id="toolbar">
            <div class="toolbar-group">
                <input type="text" id="searchInput" class="toolbar-search" placeholder="Search events...">
            </div>
            <div class="toolbar-group">
                <select id="categoryFilter" class="toolbar-select">
                    <option value="">All Categories</option>
"""
    
    for category in categories:
        html += f'                    <option value="{category}">{category}</option>\n'
    
    html += f"""                </select>
            </div>
            <div class="toolbar-group seg-group" role="group" aria-label="Significance filter">
                <button class="seg-btn" data-significance="1">All</button>
                <button class="seg-btn active" data-significance="2">Notable</button>
                <button class="seg-btn" data-significance="3">Landmarks</button>
            </div>
            <div class="toolbar-stats">
                <span id="eventCount">{len(events)}</span> events
                <span class="timeline-version" aria-hidden="true"> · v.{version}</span>
            </div>
        </div>

        <div class="parallax-bg">
            <img src="data/FranklinPark_Logo_blue.png" alt="" class="bg-logo" id="parallaxLogo">
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
            html += f'                                <span class="event-category" data-category="{event["category"]}">{event["category"]}</span>\n'
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
    
    <script src="timeline.js"></script>
</body>
</html>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Generated HTML timeline with {len(events)} events")
    print(f"Output: {output_path}")

def main() -> None:
    """Main entry point for timeline generation."""
    # Paths
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "AI_Timeline_1940-2025.csv"
    output_path = project_root / "index.html"
    
    # Create directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Version: increment on each run
        version = get_and_increment_version(project_root)
        print(f"Timeline version: {version}")
        
        # Read and generate
        print(f"Reading CSV from {csv_path}...")
        events = read_csv_data(csv_path)
        print(f"Found {len(events)} events")
        
        print(f"Generating HTML timeline...")
        generate_html(events, output_path, version=version)
        print(f"✓ Successfully generated HTML timeline with {len(events)} events")
        print(f"✓ Output: {output_path}")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("Please ensure the CSV file exists in the data/ directory")
        return
    except ValueError as e:
        print(f"✗ Error: {e}")
        return
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
