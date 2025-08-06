import pandas as pd
import requests
import time
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_geocode_single_stop(stop_name: str):
    """Debug a single stop to see what's happening"""
    print(f"\n{'='*60}")
    print(f"DEBUG GEOCODING: {stop_name}")
    print(f"{'='*60}")
    
    # Clean the name
    cleaned_name = stop_name.replace('P.S', 'Police Station').replace('O.T', 'Over Bridge')
    print(f"Cleaned name: {cleaned_name}")
    
    # Try different query variations
    queries = [
        f"{stop_name} bus stop Chennai Tamil Nadu",
        f"{cleaned_name} Chennai Tamil Nadu India",
        f"{stop_name} Chennai",
        stop_name
    ]
    
    session = requests.Session()
    session.headers.update({'User-Agent': 'DebugGeocoder/1.0'})
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': query,
                'format': 'json',
                'limit': 3,
                'countrycodes': 'IN',
                'addressdetails': 1
            }
            
            response = session.get(url, params=params, timeout=10)
            results = response.json()
            
            print(f"Found {len(results)} results:")
            for j, result in enumerate(results, 1):
                lat, lon = result['lat'], result['lon']
                display_name = result['display_name']
                print(f"  {j}. {display_name}")
                print(f"     Coordinates: {lat}, {lon}")
            
            if results:
                print(f"✓ Would use first result: {results[0]['display_name']}")
                return
                
        except Exception as e:
            print(f"✗ Query failed: {e}")
        
        time.sleep(1)
    
    print("✗ No results found for any query")

# Test a few stops from your data
test_stops = ["THIRUVOTRIYUR", "ANNA NAGAR", "ROYAPURAM P.S", "POONAMALLEE"]

for stop in test_stops:
    debug_geocode_single_stop(stop)
    print("\n" + "="*80)