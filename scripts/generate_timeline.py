#!/usr/bin/env python3
"""
Generate interactive HTML timeline from YAML data
"""
from html import escape
import yaml
from pathlib import Path
from typing import List, Dict

# Valid category enum
VALID_CATEGORIES = {
    'Research Breakthroughs',
    'Foundation Models',
    'AI Products',
    'Generative AI',
    'Enterprise & Developer Tools',
    'Hardware & Infrastructure',
    'Autonomous Systems',
    'Governance & Society'
}

def read_yaml_data(yaml_path: Path) -> List[Dict]:
    """
    Read timeline data from YAML file.

    Args:
        yaml_path: Path to the YAML file

    Returns:
        List of event dictionaries

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If YAML structure is invalid
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            events = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parse error: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"YAML file encoding error: {e}. Please ensure the file is UTF-8 encoded.")

    if not events:
        raise ValueError("No events found in YAML file")

    # Validate and normalize each event
    validated_events = []
    for idx, event in enumerate(events, start=1):
        try:
            # Check required fields
            if not event.get('year'):
                print(f"Warning: Event {idx} has no year, skipping")
                continue
            if not event.get('event'):
                print(f"Warning: Event {idx} has no event description, skipping")
                continue
            if not event.get('people'):
                print(f"Warning: Event {idx} has no people/organizations, skipping")
                continue
            if not event.get('category'):
                print(f"Warning: Event {idx} has no category, skipping")
                continue

            # Validate category
            category = str(event['category']).strip()
            if category not in VALID_CATEGORIES:
                print(f"Warning: Event {idx} has invalid category '{category}'. Valid categories: {', '.join(sorted(VALID_CATEGORIES))}")
                continue

            # Normalize event
            normalized = {
                'year': int(event['year']),
                'event': str(event['event']).strip(),
                'people': str(event['people']).strip(),
                'category': category,
                'source': str(event.get('source', '')).strip(),
                'link': str(event.get('link', '')).strip() if event.get('link') else '',
                'eventWeight': str(event.get('weight', '')).strip() if event.get('weight') else ''
            }

            validated_events.append(normalized)
        except Exception as e:
            print(f"Warning: Error processing event {idx}: {e}")
            continue

    if not validated_events:
        raise ValueError("No valid events found in YAML file after validation")

    # Sort by year
    validated_events.sort(key=lambda x: x['year'])
    return validated_events

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
    <title>AI &amp; ML Timeline ({min_year}–{max_year})</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <div class="toolbar" id="toolbar">
            <div class="toolbar-title">
                <span class="toolbar-title-text">AI &amp; ML Timeline</span>
                <span class="toolbar-title-sub">{min_year} – {max_year}</span>
            </div>
            <div class="toolbar-divider"></div>
            <div class="search-wrap">
                <input type="text" id="searchInput" class="toolbar-search" placeholder="Search events...">
            </div>
            <div class="toolbar-group">
                <select id="categoryFilter" class="toolbar-select">
                    <option value="">All Categories</option>
"""
    
    for category in categories:
        escaped_category = escape(category)
        html += f'                    <option value="{escaped_category}">{escaped_category}</option>\n'
    
    html += f"""                </select>
            </div>
            <div class="toolbar-divider"></div>
            <div class="toolbar-group seg-group" role="group" aria-label="Significance filter">
                <button class="seg-btn active" data-significance="1">All</button>
                <button class="seg-btn" data-significance="2">Notable</button>
                <button class="seg-btn" data-significance="3">Landmarks</button>
            </div>
            <div class="toolbar-divider"></div>
            <div class="toolbar-stats">
                <span id="eventCount">{len(events)}</span> events
                <span class="timeline-version" aria-hidden="true"> · v.{version}</span>
            </div>
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
            weight_attr = f' data-weight="{escape(event["eventWeight"])}"' if event['eventWeight'] else ''
            # Assign level (0, 1, 2) to alternate vertical positions
            level = i % 3
            escaped_category = escape(event["category"])
            escaped_event = escape(event["event"])
            escaped_people = escape(event["people"])
            escaped_link = escape(event["link"])

            html += f'                    <div class="event-card level-{level}" data-category="{escaped_category}"{weight_attr}>\n'
            html += f'                        <div class="event-content">\n'
            html += f'                            <h3 class="event-title">{escaped_event}</h3>\n'

            if event['people']:
                html += f'                            <p class="event-people">{escaped_people}</p>\n'

            html += f'                            <div class="event-meta">\n'
            html += f'                                <span class="event-category" data-category="{escaped_category}">{escaped_category}</span>\n'
            html += '                            </div>\n'

            if event['link']:
                html += f'                            <div class="event-link">\n'
                html += f'                                <a href="{escaped_link}" target="_blank" rel="noopener noreferrer">Details →</a>\n'
                html += '                            </div>\n'

            html += '                        </div>\n'
            html += '                    </div>\n'
        
        html += '                </div>\n'
        html += '            </div>\n'
    
    html += """            </div>
        </div>
        <div class="timeline-scrubber-wrap" aria-label="Timeline position">
            <div class="scrubber-year-label" id="scrubberYearLabel"></div>
            <input type="range" id="timelineScrubber" class="timeline-scrubber" min="0" max="100" value="0" step="0.1" title="Drag to scroll timeline">
        </div>
    </div>

    <a href="https://franklinparkllc.github.io/AI-model-basics/" target="_blank" rel="noopener noreferrer" class="model-basics-fab" title="AI Models How-To">
        <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
            <polyline points="15 3 21 3 21 9"/>
            <line x1="10" y1="14" x2="21" y2="3"/>
        </svg>
        <span>AI Models How-To</span>
    </a>

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
    yaml_path = project_root / "data" / "timeline.yaml"
    output_path = project_root / "index.html"

    # Create directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Version: increment on each run
        version = get_and_increment_version(project_root)
        print(f"Timeline version: {version}")

        # Read and generate
        print(f"Reading YAML from {yaml_path}...")
        events = read_yaml_data(yaml_path)
        print(f"Found {len(events)} events")

        print(f"Generating HTML timeline...")
        generate_html(events, output_path, version=version)
        print(f"✓ Successfully generated HTML timeline with {len(events)} events")
        print(f"✓ Output: {output_path}")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("Please ensure the YAML file exists in the data/ directory")
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
