#!/usr/bin/env python3
"""
Normalize the timeline CSV so Link and Event Weight columns are properly quoted.
This prevents URLs containing commas from bleeding into the Event Weight column.
"""
import csv
from pathlib import Path
from typing import List

def normalize_csv(csv_path: Path) -> None:
    """
    Normalize CSV file by properly quoting columns to prevent comma bleed.
    
    Args:
        csv_path: Path to the CSV file to normalize
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV structure is invalid
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    rows: List[List[str]] = []
    fixed_count = 0
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError("CSV file appears to be empty")
            
            expected_cols = len(header)
            if expected_cols < 6:
                raise ValueError(f"CSV file has {expected_cols} columns, expected at least 6")
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                original_len = len(row)
                
                # If we got too many columns, link likely bled: merge Link (col 5) + extras into Link, last is Weight
                if len(row) > expected_cols:
                    link_parts = row[5:expected_cols - 1]  # everything between Link and final Weight
                    weight = row[expected_cols - 1]
                    row = row[:5] + [','.join(link_parts)] + [weight]
                    fixed_count += 1
                    print(f"Fixed row {row_num}: merged {original_len} columns into {len(row)}")
                elif len(row) < expected_cols and len(row) > 5:
                    # Weight might be missing; pad
                    while len(row) < expected_cols:
                        row.append('')
                    fixed_count += 1
                    print(f"Fixed row {row_num}: padded from {original_len} to {len(row)} columns")
                elif len(row) != expected_cols:
                    print(f"Warning: Row {row_num} has {len(row)} columns, expected {expected_cols}")
                
                rows.append(row)
    except UnicodeDecodeError as e:
        raise ValueError(f"CSV file encoding error: {e}. Please ensure the file is UTF-8 encoded.")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    # Write back with QUOTE_NONNUMERIC so Link and other text fields are always quoted
    try:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(header)
            writer.writerows(rows)
    except Exception as e:
        raise ValueError(f"Error writing CSV file: {e}")

    print(f"✓ Normalized CSV: Link and text columns are now quoted to prevent comma bleed.")
    if fixed_count > 0:
        print(f"✓ Fixed {fixed_count} row(s) with column bleed issues")

def main() -> None:
    """Main entry point for CSV normalization."""
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "AI_Timeline_1940-2025.csv"
    
    try:
        normalize_csv(csv_path)
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("Please ensure the CSV file exists in the data/ directory")
    except ValueError as e:
        print(f"✗ Error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
