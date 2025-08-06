import pandas as pd
import requests
import time
import logging
from typing import Dict, List, Optional
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChennaiBusStopGeocoder:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ChennaiBusGeocoder/1.0 (educational purpose)'
        })
        
        # Chennai-specific name corrections
        self.name_corrections = {
            'THIRUVOTRIYUR': 'TIRUVOTTIYUR',
            'THRUVOTRIYUR': 'TIRUVOTTIYUR', 
            'ROYAPURAM P.S': 'ROYAPURAM',
            'CLIVE BATTERY': 'CLIVE BATTERY ROYAPURAM',
            'PARRYS': 'PARRYS CORNER',
            'M.G.R.CENTRAL': 'CHENNAI CENTRAL',
            'DASAPRAKASH': 'DASAPRAKASH EGMORE',
            'TAYLORS ROAD': 'TAYLOR ROAD KILPAUK',
            'AMINJIKARAI': 'AMINJIKARAI',
            'NADUVANKARAI': 'NADUVANKARAI',
            'ARUMBAKKAM': 'ARUMBAKKAM',
            'NERKUNDRAM': 'NERKUNDRAM',
            'MADURAVOYAL': 'MADURAVOYAL',
            'VAANAGARAM': 'VANAGARAM',
            'VELAPPANCHAVADI': 'VELAPPANCHAVADI',
            'KUMUNANCHAVADI': 'KUNDRATHUR',
            'POONAMALLEE': 'POONAMALLEE'
        }
        
        # Alternative search terms for difficult stops
        self.alternative_searches = {
            'THIRUVOTRIYUR': ['TIRUVOTTIYUR', 'TIRUVOTTIYUR CHENNAI', 'TIRUVOTTIYUR TEMPLE'],
            'ROYAPURAM P.S': ['ROYAPURAM', 'ROYAPURAM CHENNAI', 'ROYAPURAM STATION'],
            'CLIVE BATTERY': ['CLIVE BATTERY ROYAPURAM', 'CLIVE BATTERY CHENNAI'],
            'M.G.R.CENTRAL': ['CHENNAI CENTRAL', 'CENTRAL RAILWAY STATION CHENNAI', 'MGR CENTRAL STATION'],
            'DASAPRAKASH': ['EGMORE', 'EGMORE CHENNAI', 'DASAPRAKASH EGMORE'],
            'PARRYS': ['PARRYS CORNER', 'PARRYS CORNER CHENNAI', 'PARRY CORNER']
        }
    
    def get_corrected_name(self, stop_name: str) -> str:
        """Get corrected name for known problematic stops"""
        return self.name_corrections.get(stop_name, stop_name)
    
    def get_search_variations(self, stop_name: str) -> List[str]:
        """Generate multiple search variations for a stop"""
        corrected_name = self.get_corrected_name(stop_name)
        
        # Start with alternative searches if available
        if stop_name in self.alternative_searches:
            variations = self.alternative_searches[stop_name].copy()
        else:
            variations = [corrected_name]
        
        # Add standard variations
        base_variations = [
            f"{corrected_name} bus stop Chennai",
            f"{corrected_name} Chennai Tamil Nadu",
            f"{corrected_name} Chennai",
            f"{stop_name} Chennai",  # Original name too
            corrected_name,
            stop_name
        ]
        
        # Combine and remove duplicates
        all_variations = variations + base_variations
        return list(dict.fromkeys(all_variations))  # Preserves order, removes duplicates
    
    def search_nominatim(self, query: str) -> List[Dict]:
        """Search Nominatim with a single query"""
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': query,
                'format': 'json',
                'limit': 5,
                'countrycodes': 'IN',
                'addressdetails': 1,
                'extratags': 1,
                'viewbox': '80.0,12.8,80.3,13.2',  # Chennai bounding box
                'bounded': 1
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            results = response.json()
            
            # Filter for Chennai results
            chennai_results = []
            for result in results:
                display_name = result.get('display_name', '').lower()
                address = result.get('address', {})
                
                # Check if it's in Chennai/Tamil Nadu
                if ('chennai' in display_name or 
                    'tamil nadu' in display_name or
                    address.get('city') == 'Chennai' or
                    address.get('state') == 'Tamil Nadu'):
                    
                    chennai_results.append({
                        'lat': float(result['lat']),
                        'lon': float(result['lon']),
                        'display_name': result['display_name'],
                        'address': address,
                        'type': result.get('type', ''),
                        'class': result.get('class', '')
                    })
            
            return chennai_results
            
        except Exception as e:
            logger.debug(f"Search failed for query '{query}': {e}")
            return []
    
    def geocode_stop(self, stop_name: str, verbose: bool = False) -> Dict:
        """Geocode a single bus stop with multiple strategies"""
        if verbose:
            print(f"\n🔍 Geocoding: {stop_name}")
        
        search_variations = self.get_search_variations(stop_name)
        
        if verbose:
            print(f"   Trying {len(search_variations)} search variations...")
        
        best_result = None
        
        for i, query in enumerate(search_variations):
            if verbose:
                print(f"   Query {i+1}: {query}")
            
            results = self.search_nominatim(query)
            
            if results:
                result = results[0]  # Take the best result
                confidence = 'high' if i < 3 else 'medium' if i < 6 else 'low'
                
                if verbose:
                    print(f"   ✅ Found: {result['display_name']}")
                
                return {
                    'latitude': result['lat'],
                    'longitude': result['lon'],
                    'full_name': result['display_name'],
                    'confidence': confidence,
                    'source': 'nominatim',
                    'query_used': query,
                    'address': result.get('address', {})
                }
            
            if verbose:
                print(f"   ❌ No results")
            
            # Rate limiting
            time.sleep(0.8)
        
        if verbose:
            print(f"   ⚠️  No coordinates found for {stop_name}")
        
        return {
            'latitude': None,
            'longitude': None,
            'full_name': None,
            'confidence': 'none',
            'source': 'not_found',
            'query_used': None,
            'address': {}
        }
    
    def process_csv(self, input_file: str, output_file: str = None, verbose: bool = True) -> pd.DataFrame:
        """Process the entire CSV file"""
        df = pd.read_csv(input_file)
        unique_stops = df['stop_name'].unique()
        
        print(f"🚌 Processing {len(unique_stops)} unique bus stops from Chennai MTC...")
        print(f"📍 Using improved Chennai-specific geocoding strategies")
        
        results = {}
        
        for i, stop_name in enumerate(unique_stops, 1):
            print(f"\n[{i}/{len(unique_stops)}] Processing: {stop_name}")
            
            result = self.geocode_stop(stop_name, verbose=False)
            results[stop_name] = result
            
            # Show immediate result
            if result['latitude']:
                print(f"    ✅ Found coordinates: {result['latitude']:.6f}, {result['longitude']:.6f}")
                print(f"    📍 Location: {result['full_name']}")
            else:
                print(f"    ❌ No coordinates found")
            
            # Progress summary every 5 stops
            if i % 5 == 0:
                found = sum(1 for r in results.values() if r['latitude'] is not None)
                print(f"\n📊 Progress: {found}/{i} stops geocoded ({found/i*100:.1f}% success rate)")
            
            time.sleep(1.2)  # Rate limiting
        
        # Add results to dataframe
        df['latitude'] = df['stop_name'].map(lambda x: results[x]['latitude'])
        df['longitude'] = df['stop_name'].map(lambda x: results[x]['longitude'])
        df['full_location_name'] = df['stop_name'].map(lambda x: results[x]['full_name'])
        df['geocoding_confidence'] = df['stop_name'].map(lambda x: results[x]['confidence'])
        df['geocoding_source'] = df['stop_name'].map(lambda x: results[x]['source'])
        df['query_used'] = df['stop_name'].map(lambda x: results[x]['query_used'])
        
        # Generate summary
        total_stops = len(unique_stops)
        found_coords = sum(1 for r in results.values() if r['latitude'] is not None)
        
        print(f"\n{'='*60}")
        print(f"🎯 FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Total unique stops: {total_stops}")
        print(f"Successfully geocoded: {found_coords}")
        print(f"Success rate: {found_coords/total_stops*100:.1f}%")
        
        # Confidence breakdown
        confidence_counts = {}
        for result in results.values():
            conf = result['confidence']
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        print(f"\n📈 Confidence levels:")
        for conf, count in confidence_counts.items():
            emoji = "🎯" if conf == "high" else "🎲" if conf == "medium" else "⚠️" if conf == "low" else "❌"
            print(f"   {emoji} {conf.capitalize()}: {count} stops ({count/total_stops*100:.1f}%)")
        
        # Save results
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\n💾 Results saved to: {output_file}")
        
        return df

def test_single_stops():
    """Test the problematic stops individually"""
    geocoder = ChennaiBusStopGeocoder()
    test_stops = ["THIRUVOTRIYUR", "ANNA NAGAR", "ROYAPURAM P.S", "POONAMALLEE", "CLIVE BATTERY"]
    
    print("🧪 TESTING INDIVIDUAL STOPS")
    print("="*50)
    
    for stop in test_stops:
        result = geocoder.geocode_stop(stop, verbose=True)
        print(f"Result: {result}")
        print("-" * 50)

def main():
    """Main function"""
    # Test individual stops first
    print("First, let's test the problematic stops:")
    test_single_stops()
    
    print("\n" + "="*80)
    print("Now processing the full CSV file:")
    
    # Process the full file
    geocoder = ChennaiBusStopGeocoder()
    
    input_file = "../output/mtc_all_stops_20250805_184547.csv"
    output_file = "../output/mtc_all_stops_20250805_184547_with_coordinates.csv"

    result_df = geocoder.process_csv(input_file, output_file)
    
    # Show successful results
    successful = result_df[result_df['latitude'].notna()].drop_duplicates('stop_name')
    if not successful.empty:
        print(f"\n✅ Successfully geocoded stops:")
        for _, row in successful.iterrows():
            print(f"   • {row['stop_name']} → {row['latitude']:.6f}, {row['longitude']:.6f}")
    
    # Show failed results
    failed = result_df[result_df['latitude'].isna()]['stop_name'].unique()
    if len(failed) > 0:
        print(f"\n❌ Stops that need manual review:")
        for stop in failed:
            print(f"   • {stop}")

if __name__ == "__main__":
    main()