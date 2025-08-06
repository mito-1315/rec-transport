import pandas as pd
import requests
import time
import json
from typing import List, Dict, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BusStopGeocoder:
    def __init__(self, google_api_key: str = None, mapbox_api_key: str = None):
        """
        Initialize the geocoder with API keys
        
        Args:
            google_api_key: Google Maps API key (optional but recommended)
            mapbox_api_key: Mapbox API key (optional but recommended)
        """
        self.google_api_key = google_api_key
        self.mapbox_api_key = mapbox_api_key
        self.session = requests.Session()
        
    def search_nominatim(self, stop_name: str, city: str = "Chennai") -> List[Dict]:
        """
        Search for bus stops using OpenStreetMap Nominatim (free service)
        """
        try:
            # Search specifically for bus stops
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': f'{stop_name} bus stop {city}',
                'format': 'json',
                'limit': 5,
                'countrycodes': 'IN',
                'amenity': 'bus_station',
                'extratags': 1,
                'addressdetails': 1
            }
            
            headers = {
                'User-Agent': 'BusStopGeocoder/1.0 (contact@example.com)'
            }
            
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            results = response.json()
            
            # Filter and format results
            bus_stops = []
            for result in results:
                # Check if it's actually a bus stop/station
                if any(keyword in result.get('display_name', '').lower() 
                      for keyword in ['bus', 'stop', 'station', 'depot']):
                    bus_stops.append({
                        'name': result.get('display_name', ''),
                        'lat': float(result['lat']),
                        'lon': float(result['lon']),
                        'source': 'nominatim',
                        'place_id': result.get('place_id'),
                        'type': result.get('type', 'unknown')
                    })
            
            return bus_stops
            
        except Exception as e:
            logger.error(f"Nominatim search failed for {stop_name}: {e}")
            return []
    
    def search_google_places(self, stop_name: str, city: str = "Chennai") -> List[Dict]:
        """
        Search for bus stops using Google Places API
        """
        if not self.google_api_key:
            return []
            
        try:
            url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            params = {
                'query': f'{stop_name} bus stop {city}',
                'key': self.google_api_key,
                'type': 'bus_station',
                'location': '13.0827,80.2707',  # Chennai coordinates
                'radius': 50000  # 50km radius
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] != 'OK':
                logger.warning(f"Google Places API returned status: {data['status']}")
                return []
            
            bus_stops = []
            for result in data.get('results', []):
                # Check if it's a bus stop/station
                types = result.get('types', [])
                if any(t in ['bus_station', 'transit_station'] for t in types):
                    location = result['geometry']['location']
                    bus_stops.append({
                        'name': result['name'],
                        'lat': location['lat'],
                        'lon': location['lng'],
                        'source': 'google_places',
                        'place_id': result['place_id'],
                        'rating': result.get('rating'),
                        'types': types
                    })
            
            return bus_stops
            
        except Exception as e:
            logger.error(f"Google Places search failed for {stop_name}: {e}")
            return []
    
    def search_mapbox(self, stop_name: str, city: str = "Chennai") -> List[Dict]:
        """
        Search for bus stops using Mapbox Geocoding API
        """
        if not self.mapbox_api_key:
            return []
            
        try:
            query = f'{stop_name} bus stop {city}'
            url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{query}.json"
            params = {
                'access_token': self.mapbox_api_key,
                'country': 'IN',
                'proximity': '80.2707,13.0827',  # Chennai (lon, lat)
                'types': 'poi',
                'limit': 5
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            bus_stops = []
            
            for feature in data.get('features', []):
                # Check if it's related to transportation
                categories = feature.get('properties', {}).get('category', '')
                if 'transport' in categories.lower() or 'bus' in feature['text'].lower():
                    coordinates = feature['geometry']['coordinates']
                    bus_stops.append({
                        'name': feature['place_name'],
                        'lat': coordinates[1],
                        'lon': coordinates[0],
                        'source': 'mapbox',
                        'category': categories
                    })
            
            return bus_stops
            
        except Exception as e:
            logger.error(f"Mapbox search failed for {stop_name}: {e}")
            return []
    
    def validate_bus_stop(self, lat: float, lon: float, stop_name: str) -> bool:
        """
        Validate if the coordinates actually correspond to a bus stop
        """
        try:
            # Use reverse geocoding to check if location is near a bus stop
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                'lat': lat,
                'lon': lon,
                'format': 'json',
                'zoom': 18,
                'extratags': 1
            }
            
            headers = {
                'User-Agent': 'BusStopGeocoder/1.0 (contact@example.com)'
            }
            
            response = self.session.get(url, params=params, headers=headers, timeout=5)
            response.raise_for_status()
            
            result = response.json()
            
            # Check if the location is near transportation infrastructure
            display_name = result.get('display_name', '').lower()
            amenity = result.get('extratags', {}).get('amenity', '').lower()
            
            return any(keyword in display_name for keyword in ['bus', 'stop', 'station', 'transport']) or \
                   amenity in ['bus_station', 'bus_stop']
                   
        except Exception as e:
            logger.debug(f"Validation failed for {stop_name}: {e}")
            return True  # Assume valid if validation fails
    
    def geocode_bus_stop(self, stop_name: str, city: str = "Chennai") -> List[Dict]:
        """
        Get coordinates for a bus stop using multiple sources
        """
        all_results = []
        
        # Search using all available services
        nominatim_results = self.search_nominatim(stop_name, city)
        google_results = self.search_google_places(stop_name, city)
        mapbox_results = self.search_mapbox(stop_name, city)
        
        all_results.extend(nominatim_results)
        all_results.extend(google_results)
        all_results.extend(mapbox_results)
        
        # Remove duplicates based on proximity (within 100 meters)
        unique_results = []
        for result in all_results:
            is_duplicate = False
            for unique in unique_results:
                distance = self.calculate_distance(
                    result['lat'], result['lon'],
                    unique['lat'], unique['lon']
                )
                if distance < 0.1:  # Less than 100 meters
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # Validate the result
                if self.validate_bus_stop(result['lat'], result['lon'], stop_name):
                    unique_results.append(result)
        
        return unique_results
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points in kilometers using Haversine formula
        """
        import math
        
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def process_csv(self, csv_file: str, output_file: str = None, max_workers: int = 5) -> pd.DataFrame:
        """
        Process the entire CSV file and get coordinates for all bus stops
        """
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Get unique stop names
        unique_stops = df['stop_name'].unique()
        
        results = []
        
        def process_stop(stop_name):
            logger.info(f"Processing: {stop_name}")
            coordinates = self.geocode_bus_stop(stop_name)
            time.sleep(1)  # Rate limiting
            return stop_name, coordinates
        
        # Process stops with threading for better performance
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stop = {executor.submit(process_stop, stop): stop 
                             for stop in unique_stops}
            
            for future in as_completed(future_to_stop):
                stop_name, coordinates = future.result()
                
                if coordinates:
                    for coord in coordinates:
                        results.append({
                            'stop_name': stop_name,
                            'latitude': coord['lat'],
                            'longitude': coord['lon'],
                            'source': coord['source'],
                            'full_name': coord['name'],
                            'confidence': 'high' if len(coordinates) == 1 else 'medium'
                        })
                else:
                    # Add entry with no coordinates found
                    results.append({
                        'stop_name': stop_name,
                        'latitude': None,
                        'longitude': None,
                        'source': 'not_found',
                        'full_name': None,
                        'confidence': 'none'
                    })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save to file if specified
        if output_file:
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        
        return results_df

# Usage example and main function
def main():
    """
    Main function to demonstrate usage
    """
    # Initialize geocoder (add your API keys here for better results)
    geocoder = BusStopGeocoder(
        google_api_key="AIzaSyBlfqs5K9HEe9c1Eu5bjPXXjr8Hz2mbTZE",  # Optional but recommended
        mapbox_api_key="pk.eyJ1IjoibWl0bzEzMTUiLCJhIjoiY21kZmwyN29tMGVrbTJsc2QxdzMzYWE2MyJ9.LJJnphCN0ItrPWM2kts9yg"   # Optional but recommended
    )
    
    # Process the CSV file
    #csv_file = "../output/mtc_all_stops_20250805_184547.csv"
    csv_file = "../output/mtc_test_route.csv"
    output_file = "../output/bus_stops_with_coordinates.csv"
    
    logger.info("Starting geocoding process...")
    results_df = geocoder.process_csv(csv_file, output_file)
    
    # Display summary
    total_stops = len(results_df['stop_name'].unique())
    found_coords = len(results_df[results_df['latitude'].notna()])
    
    print(f"\nSummary:")
    print(f"Total unique stops: {total_stops}")
    print(f"Coordinates found: {found_coords}")
    print(f"Success rate: {found_coords/total_stops*100:.1f}%")
    
    # Show sample results
    print(f"\nSample results:")
    print(results_df.head(10))

if __name__ == "__main__":
    main()