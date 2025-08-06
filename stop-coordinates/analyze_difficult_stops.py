import json
import pandas as pd
import re
from collections import Counter, defaultdict

def analyze_missing_stops(json_file: str):
    """Analyze the missing stops to understand patterns"""
    
    with open(json_file, 'r') as f:
        missing_stops = json.load(f)
    
    print("🔍 Analysis of Missing Chennai Bus Stops")
    print("=" * 50)
    print(f"Total missing stops: {len(missing_stops)}")
    
    # Pattern analysis
    patterns = {
        'has_abbreviations': 0,
        'has_junction': 0,
        'has_road': 0,
        'has_temple': 0,
        'has_hospital': 0,
        'has_school_college': 0,
        'has_nagar_colony': 0,
        'has_numbers': 0,
        'has_dots': 0,
        'very_long': 0,
        'has_parentheses': 0
    }
    
    abbreviations = Counter()
    common_words = Counter()
    area_indicators = Counter()
    
    for stop in missing_stops:
        upper_stop = stop.upper()
        
        # Pattern detection
        if re.search(r'\b[A-Z]{2,4}\.?\b', stop):
            patterns['has_abbreviations'] += 1
            abbrevs = re.findall(r'\b[A-Z]{2,4}\.?\b', stop)
            abbreviations.update(abbrevs)
        
        if 'JN' in upper_stop or 'JUNCTION' in upper_stop:
            patterns['has_junction'] += 1
        
        if 'RD' in upper_stop or 'ROAD' in upper_stop:
            patterns['has_road'] += 1
        
        if any(word in upper_stop for word in ['TEMPLE', 'KOIL', 'KOVIL', 'AMMAN', 'PERUMAL']):
            patterns['has_temple'] += 1
        
        if any(word in upper_stop for word in ['HOSPITAL', 'HOSP', 'MEDICAL']):
            patterns['has_hospital'] += 1
        
        if any(word in upper_stop for word in ['SCHOOL', 'COLLEGE', 'UNIV']):
            patterns['has_school_college'] += 1
        
        if any(word in upper_stop for word in ['NAGAR', 'COLONY']):
            patterns['has_nagar_colony'] += 1
        
        if re.search(r'\d', stop):
            patterns['has_numbers'] += 1
        
        if '.' in stop:
            patterns['has_dots'] += 1
        
        if len(stop) > 30:
            patterns['very_long'] += 1
        
        if '(' in stop or ')' in stop:
            patterns['has_parentheses'] += 1
        
        # Word frequency
        words = re.findall(r'\b[A-Z]{3,}\b', upper_stop)
        common_words.update(words)
        
        # Area indicators
        area_words = ['NAGAR', 'COLONY', 'PURAM', 'PATTU', 'KUPPAM', 'CHAVADI', 'CHERY', 'PALAYAM']
        for area_word in area_words:
            if area_word in upper_stop:
                area_indicators[area_word] += 1
    
    print("\n📊 Pattern Analysis:")
    for pattern, count in patterns.items():
        percentage = (count / len(missing_stops)) * 100
        print(f"   • {pattern.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    print(f"\n🔤 Top 15 Abbreviations:")
    for abbrev, count in abbreviations.most_common(15):
        print(f"   • {abbrev}: {count} times")
    
    print(f"\n📍 Top 15 Common Words:")
    for word, count in common_words.most_common(15):
        if len(word) > 2:
            print(f"   • {word}: {count} times")
    
    print(f"\n🏘️  Area Indicators:")
    for area, count in area_indicators.most_common():
        print(f"   • {area}: {count} times")
    
    # Difficulty categories
    print(f"\n🎯 Difficulty Categories:")
    
    very_difficult = []
    difficult = []
    moderate = []
    
    for stop in missing_stops:
        upper_stop = stop.upper()
        difficulty_score = 0
        
        # Increase difficulty for various factors
        if len(re.findall(r'\b[A-Z]{2,4}\.?\b', stop)) > 2:
            difficulty_score += 3
        elif re.search(r'\b[A-Z]{2,4}\.?\b', stop):
            difficulty_score += 1
        
        if len(stop) > 35:
            difficulty_score += 2
        elif len(stop) > 25:
            difficulty_score += 1
        
        if stop.count('.') > 3:
            difficulty_score += 2
        elif '.' in stop:
            difficulty_score += 1
        
        if '(' in stop:
            difficulty_score += 1
        
        if re.search(r'\d', stop):
            difficulty_score += 1
        
        # Uncommon words
        words = upper_stop.split()
        if any(len(word) > 12 for word in words):
            difficulty_score += 1
        
        if difficulty_score >= 6:
            very_difficult.append(stop)
        elif difficulty_score >= 3:
            difficult.append(stop)
        else:
            moderate.append(stop)
    
    print(f"   🔴 Very Difficult: {len(very_difficult)} stops")
    print(f"   🟡 Difficult: {len(difficult)} stops") 
    print(f"   🟢 Moderate: {len(moderate)} stops")
    
    print(f"\n🔴 Sample Very Difficult Stops:")
    for stop in very_difficult[:10]:
        print(f"   • {stop}")
    
    print(f"\n🟡 Sample Difficult Stops:")
    for stop in difficult[:10]:
        print(f"   • {stop}")
    
    return {
        'patterns': patterns,
        'abbreviations': abbreviations,
        'common_words': common_words,
        'area_indicators': area_indicators,
        'difficulty_categories': {
            'very_difficult': very_difficult,
            'difficult': difficult,
            'moderate': moderate
        }
    }

if __name__ == "__main__":
    analysis = analyze_missing_stops("../output/mtc_no_coords.json")