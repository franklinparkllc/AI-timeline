#!/usr/bin/env python3
"""
Normalize the timeline CSV so Link and Event Weight columns are properly quoted.
This prevents URLs containing commas from bleeding into the Event Weight column.
"""
import csv
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "AI_Timeline_1940-2025.csv"

    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        expected_cols = len(header)
        for i, row in enumerate(reader):
            # If we got too many columns, link likely bled: merge Link (col 5) + extras into Link, last is Weight
            if len(row) > expected_cols:
                link_parts = row[5:expected_cols - 1]  # everything between Link and final Weight
                weight = row[expected_cols - 1]
                row = row[:5] + [','.join(link_parts)] + [weight]
            elif len(row) < expected_cols and len(row) > 5:
                # Weight might be missing; pad
                while len(row) < expected_cols:
                    row.append('')
            rows.append(row)

    # Write back with QUOTE_NONNUMERIC so Link and other text fields are always quoted
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(header)
        writer.writerows(rows)

    print("Normalized CSV: Link and text columns are now quoted to prevent comma bleed.")

if __name__ == "__main__":
    main()
