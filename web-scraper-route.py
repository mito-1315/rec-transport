#!/usr/bin/env python3
"""
Web Scraper Script
A flexible web scraper that can extract HTML elements from websites.
Supports various scraping methods and element selection options.
"""

import requests
from bs4 import BeautifulSoup
import argparse
import json
import csv
import sys
from urllib.parse import urljoin, urlparse
import time
from typing import List, Dict, Optional
import re
import urllib3

# Disable SSL warnings when verification is disabled
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class WebScraper:
    def __init__(self, delay: float = 1.0, headers: Optional[Dict] = None, verify_ssl: bool = True, max_retries: int = 3):
        """
        Initialize the web scraper.
        
        Args:
            delay: Delay between requests in seconds
            headers: Custom headers for requests
            verify_ssl: Whether to verify SSL certificates
            max_retries: Maximum number of retry attempts
        """
        self.delay = delay
        self.verify_ssl = verify_ssl
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Default headers to appear more like a real browser
        default_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        if headers:
            default_headers.update(headers)
        
        self.session.headers.update(default_headers)
    
    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse a webpage with retry logic.
        
        Args:
            url: The URL to scrape
            
        Returns:
            BeautifulSoup object or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                print(f"Fetching: {url} (attempt {attempt + 1}/{self.max_retries})")
                
                response = self.session.get(
                    url, 
                    timeout=30, 
                    verify=self.verify_ssl,
                    allow_redirects=True
                )
                response.raise_for_status()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                print(f"Successfully fetched page (status: {response.status_code})")
                return soup
                
            except requests.exceptions.SSLError as e:
                print(f"SSL Error on attempt {attempt + 1}: {e}")
                if not self.verify_ssl:
                    print("SSL verification already disabled, retrying...")
                else:
                    print("Retrying with SSL verification disabled...")
                    self.verify_ssl = False
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))  # Exponential backoff
                
            except requests.exceptions.RequestException as e:
                print(f"Request Error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))
                
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay)
        
        print(f"Failed to fetch {url} after {self.max_retries} attempts")
        return None
    
    def scrape_by_tag(self, soup: BeautifulSoup, tag: str, limit: Optional[int] = None) -> List[Dict]:
        """Scrape elements by HTML tag."""
        elements = soup.find_all(tag, limit=limit)
        results = []
        
        for elem in elements:
            result = {
                'tag': elem.name,
                'text': elem.get_text(strip=True),
                'attributes': dict(elem.attrs) if elem.attrs else {},
                'html': str(elem)
            }
            results.append(result)
        
        return results
    
    def scrape_by_class(self, soup: BeautifulSoup, class_name: str, limit: Optional[int] = None) -> List[Dict]:
        """Scrape elements by CSS class."""
        elements = soup.find_all(class_=class_name, limit=limit)
        results = []
        
        for elem in elements:
            result = {
                'tag': elem.name,
                'class': class_name,
                'text': elem.get_text(strip=True),
                'attributes': dict(elem.attrs) if elem.attrs else {},
                'html': str(elem)
            }
            results.append(result)
        
        return results
    
    def scrape_by_id(self, soup: BeautifulSoup, element_id: str) -> Optional[Dict]:
        """Scrape element by ID."""
        element = soup.find(id=element_id)
        
        if element:
            return {
                'tag': element.name,
                'id': element_id,
                'text': element.get_text(strip=True),
                'attributes': dict(element.attrs) if element.attrs else {},
                'html': str(element)
            }
        return None
    
    def scrape_by_css_selector(self, soup: BeautifulSoup, selector: str, limit: Optional[int] = None) -> List[Dict]:
        """Scrape elements by CSS selector."""
        elements = soup.select(selector)
        if limit:
            elements = elements[:limit]
        
        results = []
        for elem in elements:
            result = {
                'tag': elem.name,
                'selector': selector,
                'text': elem.get_text(strip=True),
                'attributes': dict(elem.attrs) if elem.attrs else {},
                'html': str(elem)
            }
            results.append(result)
        
        return results
    
    def scrape_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract all links from the page."""
        links = soup.find_all('a', href=True)
        results = []
        
        for link in links:
            href = link['href']
            absolute_url = urljoin(base_url, href)
            
            result = {
                'text': link.get_text(strip=True),
                'href': href,
                'absolute_url': absolute_url,
                'attributes': dict(link.attrs) if link.attrs else {}
            }
            results.append(result)
        
        return results
    
    def scrape_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract all images from the page."""
        images = soup.find_all('img')
        results = []
        
        for img in images:
            src = img.get('src', '')
            if src:
                absolute_url = urljoin(base_url, src)
                
                result = {
                    'src': src,
                    'absolute_url': absolute_url,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', ''),
                    'attributes': dict(img.attrs) if img.attrs else {}
                }
                results.append(result)
        
        return results
    
    def scrape_tables(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract all tables from the page."""
        tables = soup.find_all('table')
        results = []
        
        for i, table in enumerate(tables):
            rows = []
            for row in table.find_all('tr'):
                cells = []
                for cell in row.find_all(['td', 'th']):
                    cells.append(cell.get_text(strip=True))
                if cells:
                    rows.append(cells)
            
            result = {
                'table_index': i,
                'rows': rows,
                'row_count': len(rows),
                'html': str(table)
            }
            results.append(result)
        
        return results
    
    def remove_route_num(self, route_name: str) -> str:
        """Remove route number from the route name."""
        # Assuming route numbers are at the start and followed by a space
        return re.sub(r'^\d+\s*', '', route_name).strip()

    def scrape_route_names(self, soup: BeautifulSoup, route_number: str = None) -> List[Dict]:
        """Extract route names from MTC bus website structure."""
        results = []
        
        # Find the specific div container
        route_container = soup.find('div', class_='col-md-5 col-md-offset-3')
        if not route_container:
            print("Route container div not found")
            return results
        
        # Find the ul with class "route"
        route_ul = route_container.find('ul', class_='route')
        if not route_ul:
            print("Route ul not found")
            return results
        
        # Extract all li elements and their span values
        route_items = route_ul.find_all('li')
        print(f"Found {len(route_items)} route items")
        
        for i, li in enumerate(route_items):
            span = li.find('span')
            if span:
                stop_number = span.get_text(strip=True)
                # Get the text after the span (stop name)
                full_text = li.get_text(strip=True)
                stop_name = full_text.replace(stop_number, '', 1).strip()
                
                result = {
                    'route_number': route_number or 'UNKNOWN',
                    'stop_number': stop_number,
                    'stop_name': stop_name,
                    'sequence': i + 1
                }
                results.append(result)
            else:
                # Handle li without span
                text = li.get_text(strip=True)
                if text:  # Only add if there's actual text
                    result = {
                        'route_number': route_number or 'UNKNOWN',
                        'stop_number': '',
                        'stop_name': text,
                        'sequence': i + 1
                    }
                    results.append(result)
        
        return results

def save_results(data: List[Dict], filename: str, format_type: str = 'json'):
    """Save results to file in specified format."""
    try:
        if format_type.lower() == 'json':
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format_type.lower() == 'csv' and data:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                if data:
                    # Ensure consistent field order
                    fieldnames = ['route_number', 'stop_number', 'stop_name', 'sequence']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)
        
        print(f"Results saved to {filename}")
    
    except Exception as e:
        print(f"Error saving results: {e}")

def extract_route_from_url(url: str) -> str:
    """Extract route number from URL parameters."""
    import urllib.parse as urlparse
    parsed = urlparse.urlparse(url)
    params = urlparse.parse_qs(parsed.query)
    
    route_param = params.get('selroute', [''])[0]
    return route_param if route_param else 'UNKNOWN'

def main():
    parser = argparse.ArgumentParser(description='Web Scraper - Extract HTML elements from websites')
    parser.add_argument('url', help='URL to scrape (wrap in quotes if it contains special characters)')
    parser.add_argument('--tag', help='HTML tag to scrape (e.g., h1, p, div)')
    parser.add_argument('--class', dest='class_name', help='CSS class to scrape')
    parser.add_argument('--id', help='Element ID to scrape')
    parser.add_argument('--selector', help='CSS selector to scrape')
    parser.add_argument('--links', action='store_true', help='Extract all links')
    parser.add_argument('--images', action='store_true', help='Extract all images')
    parser.add_argument('--tables', action='store_true', help='Extract all tables')
    parser.add_argument('--routes', action='store_true', help='Extract MTC bus route stops')
    parser.add_argument('--route-number', help='Specify route number manually')
    parser.add_argument('--all', action='store_true', help='Extract everything')
    parser.add_argument('--limit', type=int, help='Limit number of results')
    parser.add_argument('--output', help='Output filename (without extension)')
    parser.add_argument('--format', choices=['json', 'csv', 'both'], default='json', help='Output format')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests')
    parser.add_argument('--no-ssl-verify', action='store_true', help='Disable SSL certificate verification')
    parser.add_argument('--retries', type=int, default=3, help='Maximum number of retry attempts')
    
    args = parser.parse_args()
    
    # Validate URL
    if not args.url.startswith(('http://', 'https://')):
        print("Error: URL must start with http:// or https://")
        sys.exit(1)
    
    print(f"Scraping URL: {args.url}")
    
    # Initialize scraper
    scraper = WebScraper(
        delay=args.delay, 
        verify_ssl=not args.no_ssl_verify,
        max_retries=args.retries
    )
    
    # Fetch the page
    soup = scraper.fetch_page(args.url)
    if not soup:
        print("Failed to fetch the page")
        sys.exit(1)
    
    results = []
    
    # Extract route number from URL or use provided one
    route_number = args.route_number or extract_route_from_url(args.url)
    
    # Scrape based on arguments
    if args.tag:
        results = scraper.scrape_by_tag(soup, args.tag, args.limit)
        print(f"Found {len(results)} elements with tag '{args.tag}'")
    
    elif args.class_name:
        results = scraper.scrape_by_class(soup, args.class_name, args.limit)
        print(f"Found {len(results)} elements with class '{args.class_name}'")
    
    elif args.id:
        result = scraper.scrape_by_id(soup, args.id)
        results = [result] if result else []
        print(f"Found {len(results)} elements with id '{args.id}'")
    
    elif args.selector:
        results = scraper.scrape_by_css_selector(soup, args.selector, args.limit)
        print(f"Found {len(results)} elements with selector '{args.selector}'")
    
    elif args.links:
        results = scraper.scrape_links(soup, args.url)
        print(f"Found {len(results)} links")
    
    elif args.images:
        results = scraper.scrape_images(soup, args.url)
        print(f"Found {len(results)} images")
    
    elif args.tables:
        results = scraper.scrape_tables(soup)
        print(f"Found {len(results)} tables")
    
    elif args.routes:
        results = scraper.scrape_route_names(soup, route_number)
        print(f"Found {len(results)} route stops for route {route_number}")
    
    elif args.all:
        all_results = {
            'url': args.url,
            'title': soup.title.string if soup.title else '',
            'headings': {
                'h1': scraper.scrape_by_tag(soup, 'h1'),
                'h2': scraper.scrape_by_tag(soup, 'h2'),
                'h3': scraper.scrape_by_tag(soup, 'h3')
            },
            'paragraphs': scraper.scrape_by_tag(soup, 'p', 10),  # Limit paragraphs
            'links': scraper.scrape_links(soup, args.url),
            'images': scraper.scrape_images(soup, args.url),
            'tables': scraper.scrape_tables(soup),
            'routes': scraper.scrape_route_names(soup, route_number)
        }
        results = [all_results]
        print("Extracted all content from the page")
    
    else:
        # Default: extract route stops
        results = scraper.scrape_route_names(soup, route_number)
        print(f"Found {len(results)} route stops for route {route_number}")
    
    # Display results
    if results:
        print("\n" + "="*50)
        print("RESULTS:")
        print("="*50)
        
        # If results are route stops, display them nicely
        if isinstance(results, list) and results and 'route_number' in results[0]:
            print(f"\nRoute {results[0]['route_number']} stops:")
            for result in results[:10]:  # Show first 10
                print(f"  {result['sequence']}. Stop {result['stop_number']}: {result['stop_name']}")
            
            if len(results) > 10:
                print(f"  ... and {len(results) - 10} more stops")
        else:
            # Standard display for other types
            for i, result in enumerate(results[:5]):  # Show first 5 results
                print(f"\nResult {i+1}:")
                if isinstance(result, dict):
                    for key, value in result.items():
                        if key == 'html' and len(str(value)) > 200:
                            print(f"{key}: {str(value)[:200]}...")
                        elif isinstance(value, (list, dict)) and len(str(value)) > 200:
                            print(f"{key}: [Large data structure - {len(value) if isinstance(value, list) else len(str(value))} items]")
                        else:
                            print(f"{key}: {value}")
                print("-" * 30)
            
            if len(results) > 5:
                print(f"\n... and {len(results) - 5} more results")
    
    # Save results if requested
    if args.output:
        if args.format == 'both':
            # Save both formats
            save_results(results, f"{args.output}.json", 'json')
            save_results(results, f"{args.output}.csv", 'csv')
        else:
            # Save in specified format
            extension = '.json' if args.format == 'json' else '.csv'
            save_results(results, f"{args.output}{extension}", args.format)
    
    print(f"\nScraping completed! Found {len(results)} results.")

if __name__ == "__main__":
    # If run directly, you can also use it programmatically
    if len(sys.argv) == 1:
        # Interactive mode
        print("Web Scraper - Interactive Mode")
        print("=" * 40)
        
        url = input("Enter URL to scrape: ").strip()
        if not url:
            print("URL is required!")
            sys.exit(1)
        
        print("\nScraping options:")
        print("1. Extract by HTML tag")
        print("2. Extract by CSS class")
        print("3. Extract by element ID")
        print("4. Extract by CSS selector")
        print("5. Extract all links")
        print("6. Extract all images")
        print("7. Extract all tables")
        print("8. Extract MTC bus route stops")
        print("9. Extract everything")
        
        choice = input("\nEnter choice (1-9): ").strip()
        
        # Ask about SSL verification
        ssl_verify = input("Verify SSL certificates? (y/n, default=y): ").strip().lower()
        verify_ssl = ssl_verify != 'n'
        
        scraper = WebScraper(verify_ssl=verify_ssl)
        soup = scraper.fetch_page(url)
        
        if not soup:
            print("Failed to fetch the page")
            sys.exit(1)
        
        results = []
        route_number = extract_route_from_url(url)
        
        if choice == '1':
            tag = input("Enter HTML tag (e.g., h1, p, div): ").strip()
            results = scraper.scrape_by_tag(soup, tag)
        elif choice == '2':
            class_name = input("Enter CSS class: ").strip()
            results = scraper.scrape_by_class(soup, class_name)
        elif choice == '3':
            element_id = input("Enter element ID: ").strip()
            result = scraper.scrape_by_id(soup, element_id)
            results = [result] if result else []
        elif choice == '4':
            selector = input("Enter CSS selector: ").strip()
            results = scraper.scrape_by_css_selector(soup, selector)
        elif choice == '5':
            results = scraper.scrape_links(soup, url)
        elif choice == '6':
            results = scraper.scrape_images(soup, url)
        elif choice == '7':
            results = scraper.scrape_tables(soup)
        elif choice == '8':
            results = scraper.scrape_route_names(soup, route_number)
        elif choice == '9':
            all_results = {
                'url': url,
                'title': soup.title.string if soup.title else '',
                'links': scraper.scrape_links(soup, url),
                'images': scraper.scrape_images(soup, url),
                'tables': scraper.scrape_tables(soup),
                'routes': scraper.scrape_route_names(soup, route_number)
            }
            results = [all_results]
        
        # Display results
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results[:3]):
            print(f"\nResult {i+1}: {result}")
        
        if len(results) > 3:
            print(f"... and {len(results) - 3} more results")
    else:
        try:
            main()
        except KeyboardInterrupt:
            print("\nScraping interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            print("\nTip: If your URL contains special characters like &, ?, or =, wrap it in quotes:")
            print("python web-scraper-route.py 'your-url-here' --options")
            sys.exit(1)