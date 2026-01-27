#!/usr/bin/env python3
"""
Remap categories in the AI timeline CSV to Taxonomy 2
"""
import csv
from pathlib import Path

# Taxonomy 2 mapping
CATEGORY_MAPPING = {
    # Research Breakthroughs
    'Theory': 'Research Breakthroughs',
    'Neural Networks': 'Research Breakthroughs',
    'Neural Networks/Theory': 'Research Breakthroughs',
    'Neural Networks/Architecture': 'Research Breakthroughs',
    'Reinforcement Learning': 'Research Breakthroughs',
    'ML': 'Research Breakthroughs',
    'Games/ML': 'Research Breakthroughs',
    'Foundation': 'Research Breakthroughs',
    'Expert Systems': 'Research Breakthroughs',
    'Mathematics': 'Research Breakthroughs',
    'Science': 'Research Breakthroughs',
    'Neuromorphic': 'Research Breakthroughs',
    'Distributed Systems': 'Research Breakthroughs',
    
    # Foundation Models
    'LLMs': 'Foundation Models',
    'NLP/LLMs': 'Foundation Models',
    'LLMs/Company': 'Foundation Models',
    'LLMs/Science': 'Foundation Models',
    'Multimodal/LLMs': 'Foundation Models',
    'Reasoning/LLMs': 'Foundation Models',
    'NLP': 'Foundation Models',
    'Speech': 'Foundation Models',
    'Speech/Audio': 'Foundation Models',
    
    # AI Products
    'Applications': 'AI Products',
    'Consumer AI': 'AI Products',
    'Customization': 'AI Products',
    'Search': 'AI Products',
    'Search/Applications': 'AI Products',
    'On-device AI': 'AI Products',
    'Adoption': 'AI Products',
    'Business Model': 'AI Products',
    
    # Generative AI
    'Generative AI': 'Generative AI',
    'Generative AI/Vision': 'Generative AI',
    'Generative AI/Video': 'Generative AI',
    'ML/Generative AI': 'Generative AI',
    'Video Generation': 'Generative AI',
    'Audio/Voice': 'Generative AI',
    'Vision': 'Generative AI',
    'Multimodal': 'Generative AI',
    
    # Enterprise & Developer Tools
    'Frameworks': 'Enterprise & Developer Tools',
    'Framework': 'Enterprise & Developer Tools',
    'Code Generation': 'Enterprise & Developer Tools',
    'Developer Tools': 'Enterprise & Developer Tools',
    
    # Hardware & Infrastructure
    'Hardware': 'Hardware & Infrastructure',
    'Hardware/Autonomous': 'Hardware & Infrastructure',
    'Market/Hardware': 'Hardware & Infrastructure',
    'Infrastructure': 'Hardware & Infrastructure',
    
    # Autonomous Systems
    'Robotics': 'Autonomous Systems',
    'Autonomous Vehicles': 'Autonomous Systems',
    'Games': 'Autonomous Systems',
    'Games/RL': 'Autonomous Systems',
    'Agents': 'Autonomous Systems',
    'Deployment': 'Autonomous Systems',
    
    # Governance & Society
    'Company': 'Governance & Society',
    'Governance': 'Governance & Society',
    'Regulation': 'Governance & Society',
    'Ethics/LLMs': 'Governance & Society',
    'Ethics/Detection': 'Governance & Society',
    'Safety': 'Governance & Society',
    'Safety/Research': 'Governance & Society',
    'Funding': 'Governance & Society',
    'Reasoning': 'Governance & Society',
}

def remap_csv(input_path, output_path):
    """Remap categories in CSV file"""
    rows = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            old_category = row['Category']
            new_category = CATEGORY_MAPPING.get(old_category, old_category)
            
            # Handle special cases
            if new_category not in CATEGORY_MAPPING.values():
                print(f"Warning: Unmapped category '{old_category}' - assigning to Research Breakthroughs")
                new_category = 'Research Breakthroughs'
            
            row['Category'] = new_category
            rows.append(row)
    
    # Write updated CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Remapped {len(rows)} events")
    print(f"Output: {output_path}")
    
    # Print category distribution
    category_counts = {}
    for row in rows:
        cat = row['Category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nNew category distribution:")
    for cat in sorted(category_counts.keys()):
        print(f"  {cat}: {category_counts[cat]}")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "AI_Timeline_1940-2025.csv"
    output_path = input_path  # Overwrite the original
    
    remap_csv(input_path, output_path)
