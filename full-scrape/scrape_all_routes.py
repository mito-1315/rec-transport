#!/usr/bin/env python3
"""
MTC Bus Route Scraper - Bulk Route Information Extractor
Scrapes route stop information for all routes in simple_routes.json
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import sys
import os
from typing import List, Dict, Optional
import urllib3
from urllib.parse import urljoin

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class MTCRouteScraper:
    def __init__(self, delay: float = 2.0, max_retries: int = 3):
        """
        Initialize the MTC route scraper.
        
        Args:
            delay: Delay between requests in seconds
            max_retries: Maximum number of retry attempts
        """
        self.delay = delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.base_url = "https://mtcbus.tn.gov.in/Home/routewiseinfo"
        self.csrf_token = "7c5f6f44c6fe5925cfa767f51c4671c9"  # You may need to update this
        
        # Headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://mtcbus.tn.gov.in/Home/routewiseinfo'
        })
        
        # Disable SSL verification
        self.session.verify = False
    
    def load_routes(self, filename: str = 'simple_routes.json') -> List[str]:
        """Load route numbers from JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                routes = json.load(f)
            print(f"Loaded {len(routes)} routes from {filename}")
            return routes
        except FileNotFoundError:
            print(f"Error: {filename} not found!")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing {filename}: {e}")
            return []
    
    def get_route_url(self, route_number: str) -> str:
        """Generate URL for a specific route."""
        return f"{self.base_url}?csrf_test_name={self.csrf_token}&selroute={route_number}&submit="
    
    def fetch_route_page(self, route_number: str) -> Optional[BeautifulSoup]:
        """
        Fetch route page with retry logic.
        
        Args:
            route_number: The route number to fetch
            
        Returns:
            BeautifulSoup object or None if failed
        """
        url = self.get_route_url(route_number)
        
        for attempt in range(self.max_retries):
            try:
                print(f"Fetching route {route_number} (attempt {attempt + 1}/{self.max_retries})")
                
                response = self.session.get(url, timeout=30, allow_redirects=True)
                response.raise_for_status()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                print(f"Successfully fetched route {route_number} (status: {response.status_code})")
                return soup
                
            except requests.exceptions.RequestException as e:
                print(f"Request Error for route {route_number} on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))  # Exponential backoff
                
            except Exception as e:
                print(f"Unexpected error for route {route_number} on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay)
        
        print(f"Failed to fetch route {route_number} after {self.max_retries} attempts")
        return None
    
    def extract_route_stops(self, soup: BeautifulSoup, route_number: str) -> List[Dict]:
        """
        Extract route stop information from the page.
        
        Args:
            soup: BeautifulSoup object of the page
            route_number: The route number being scraped
            
        Returns:
            List of stop information dictionaries
        """
        stops = []
        
        try:
            # Find the route container div
            route_container = soup.find('div', class_='col-md-5 col-md-offset-3')
            if not route_container:
                print(f"Route container not found for route {route_number}")
                return stops
            
            # Find the ul with class "route"
            route_ul = route_container.find('ul', class_='route')
            if not route_ul:
                print(f"Route ul not found for route {route_number}")
                return stops
            
            # Extract all li elements
            route_items = route_ul.find_all('li')
            print(f"Found {len(route_items)} stops for route {route_number}")
            
            for i, li in enumerate(route_items):
                span = li.find('span')
                if span:
                    stop_number = span.get_text(strip=True)
                    # Get the text after the span (stop name)
                    full_text = li.get_text(strip=True)
                    stop_name = full_text.replace(stop_number, '', 1).strip()
                    
                    stop_info = {
                        'route_number': route_number,
                        'stop_number': stop_number,
                        'stop_name': stop_name,
                        'sequence': i + 1
                    }
                    stops.append(stop_info)
                else:
                    # Handle li without span
                    text = li.get_text(strip=True)
                    if text:
                        stop_info = {
                            'route_number': route_number,
                            'stop_number': '',
                            'stop_name': text,
                            'sequence': i + 1
                        }
                        stops.append(stop_info)
            
        except Exception as e:
            print(f"Error extracting stops for route {route_number}: {e}")
        
        return stops
    
    def scrape_all_routes(self, routes: List[str], start_from: int = 0) -> Dict:
        """
        Scrape all routes and return consolidated data.
        
        Args:
            routes: List of route numbers to scrape
            start_from: Index to start from (for resuming interrupted scraping)
            
        Returns:
            Dictionary containing all route data
        """
        all_routes_data = {
            'metadata': {
                'total_routes': len(routes),
                'scraped_routes': 0,
                'failed_routes': [],
                'scraping_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'routes': {}
        }
        
        successful_scrapes = 0
        failed_routes = []
        
        # Start from specified index (useful for resuming)
        routes_to_process = routes[start_from:]
        
        print(f"\nStarting to scrape {len(routes_to_process)} routes...")
        print(f"Starting from index {start_from}")
        print("="*60)
        
        for i, route_number in enumerate(routes_to_process, start_from):
            print(f"\n[{i+1}/{len(routes)}] Processing route: {route_number}")
            
            # Fetch the route page
            soup = self.fetch_route_page(route_number)
            
            if soup:
                # Extract stops for this route
                stops = self.extract_route_stops(soup, route_number)
                
                if stops:
                    all_routes_data['routes'][route_number] = {
                        'route_number': route_number,
                        'total_stops': len(stops),
                        'stops': stops,
                        'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    successful_scrapes += 1
                    print(f"✓ Successfully scraped {len(stops)} stops for route {route_number}")
                else:
                    failed_routes.append(route_number)
                    print(f"✗ No stops found for route {route_number}")
            else:
                failed_routes.append(route_number)
                print(f"✗ Failed to fetch route {route_number}")
            
            # Save progress every 10 routes
            if (i + 1) % 10 == 0:
                self.save_progress(all_routes_data, f"progress_routes_{i+1}.json")
                print(f"\n💾 Progress saved after {i+1} routes")
            
            # Delay between requests to be respectful
            if i < len(routes) - 1:  # Don't delay after the last route
                print(f"Waiting {self.delay} seconds before next request...")
                time.sleep(self.delay)
        
        # Update metadata
        all_routes_data['metadata']['scraped_routes'] = successful_scrapes
        all_routes_data['metadata']['failed_routes'] = failed_routes
        
        print(f"\n{'='*60}")
        print(f"Scraping completed!")
        print(f"Successfully scraped: {successful_scrapes}/{len(routes)} routes")
        print(f"Failed routes: {len(failed_routes)}")
        if failed_routes:
            print(f"Failed route numbers: {', '.join(failed_routes[:10])}{'...' if len(failed_routes) > 10 else ''}")
        
        return all_routes_data
    
    def save_progress(self, data: Dict, filename: str):
        """Save progress to a JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving progress: {e}")
    
    def save_results(self, data: Dict, filename: str = None):
        """Save final results to JSON file."""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"mtc_all_routes_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Final results saved to: {filename}")
            
            # Also save a summary
            summary_filename = filename.replace('.json', '_summary.json')
            summary = {
                'metadata': data['metadata'],
                'route_summary': {
                    route_num: {
                        'total_stops': info['total_stops'],
                        'first_stop': info['stops'][0]['stop_name'] if info['stops'] else '',
                        'last_stop': info['stops'][-1]['stop_name'] if info['stops'] else ''
                    }
                    for route_num, info in data['routes'].items()
                }
            }
            
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"📊 Summary saved to: {summary_filename}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def generate_csv_output(self, data: Dict, filename: str = None):
        """Generate CSV output with all stops."""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"mtc_all_stops_{timestamp}.csv"
        
        try:
            import csv
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['route_number', 'stop_number', 'stop_name', 'sequence']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write all stops from all routes
                for route_data in data['routes'].values():
                    for stop in route_data['stops']:
                        writer.writerow(stop)
            
            print(f"📊 CSV output saved to: {filename}")
            
        except Exception as e:
            print(f"Error generating CSV output: {e}")

def main():
    print("MTC Bus Route Scraper - Bulk Route Information Extractor")
    print("="*65)
    
    # Initialize scraper
    scraper = MTCRouteScraper(delay=2.0, max_retries=3)
    
    # Load routes from JSON file
    routes = scraper.load_routes('simple_routes.json')
    
    if not routes:
        print("No routes to scrape. Exiting.")
        sys.exit(1)
    
    # Ask user for confirmation
    print(f"\nFound {len(routes)} routes to scrape.")
    print("This will take approximately {:.1f} minutes.".format(len(routes) * 2.5 / 60))  # Rough estimate
    
    proceed = input("\nDo you want to proceed? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Scraping cancelled.")
        sys.exit(0)
    
    # Ask about starting point (for resuming)
    start_from = 0
    resume = input("Do you want to resume from a specific route index? (y/n): ").strip().lower()
    if resume == 'y':
        try:
            start_from = int(input("Enter the route index to start from (0-based): "))
            start_from = max(0, min(start_from, len(routes) - 1))
        except ValueError:
            print("Invalid index, starting from 0")
            start_from = 0
    
    # Start scraping
    try:
        all_data = scraper.scrape_all_routes(routes, start_from)
        
        # Save results
        scraper.save_results(all_data)
        scraper.generate_csv_output(all_data)
        
        print(f"\n🎉 Scraping completed successfully!")
        print(f"Total routes processed: {all_data['metadata']['scraped_routes']}")
        print(f"Total stops collected: {sum(len(route_data['stops']) for route_data in all_data['routes'].values())}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrupted by user!")
        print("Progress has been saved in progress files.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during scraping: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()