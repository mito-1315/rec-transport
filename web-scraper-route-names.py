import requests
from bs4 import BeautifulSoup
import ssl
import urllib3
from urllib3.exceptions import InsecureRequestWarning
import time
import json
import re

# Disable SSL warnings
urllib3.disable_warnings(InsecureRequestWarning)

def is_meaningful_route_text(text):
    """
    Check if the route text is meaningful (not just a number)
    Returns True if text contains letters or is a descriptive route name
    """
    if not text or text.strip() == '':
        return False
    
    # Skip placeholder text
    if text.lower().strip() in ['--route--', 'select route', 'choose route']:
        return False
    
    # Check if text is just a number
    if re.match(r'^\d+$', text.strip()):
        return False
    
    # Accept if text contains letters or is a combination of numbers and text
    return bool(re.search(r'[a-zA-Z]', text)) or len(text.strip()) > 3

def scrape_route_options():
    """
    Scrape all option values from the route select dropdown on mtcbus.tn.gov.in
    """
    url = "https://mtcbus.tn.gov.in/Home/routewiseinfo"
    
    # Headers to mimic a real browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    }
    
    # Create a session with SSL verification disabled
    session = requests.Session()
    session.verify = False  # Disable SSL verification
    
    # Try multiple approaches
    approaches = [
        {"description": "HTTPS with SSL verification disabled", "url": url},
        {"description": "HTTP (if available)", "url": url.replace("https://", "http://")},
    ]
    
    for approach in approaches:
        try:
            print(f"Trying: {approach['description']}")
            print(f"URL: {approach['url']}")
            
            response = session.get(
                approach['url'], 
                headers=headers, 
                timeout=30,
                allow_redirects=True
            )
            response.raise_for_status()
            
            print(f"Success! Status Code: {response.status_code}")
            print(f"Content length: {len(response.content)} bytes")
            
            # Parse the HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Debug: Print some of the page content to verify we got the right page
            title = soup.find('title')
            if title:
                print(f"Page title: {title.get_text().strip()}")
            
            # Find the div with class col-sm-3
            div_elements = soup.find_all('div', class_='col-sm-3')
            print(f"Found {len(div_elements)} div elements with class 'col-sm-3'")
            
            route_options = []
            
            # Check each div for the select element
            for i, div_element in enumerate(div_elements):
                print(f"Checking div {i+1}...")
                
                # Find the select element with name='selroute' and id='selroute'
                select_element = div_element.find('select', {'name': 'selroute', 'id': 'selroute'})
                
                if select_element:
                    print(f"Found select element in div {i+1}!")
                    
                    # Extract all option values
                    options = select_element.find_all('option')
                    print(f"Found {len(options)} options")
                    print("\nExtracting option values:")
                    print("-" * 50)
                    
                    for option in options:
                        value = option.get('value', '')
                        text = option.get_text(strip=True)
                        
                        if value or text:  # Only include non-empty options
                            route_options.append({
                                'value': value,
                                'text': text
                            })
                            print(f"Value: '{value}' | Text: '{text}'")
                    
                    return route_options
            
            # If we didn't find the specific select, let's look for any select elements
            print("Specific select not found. Looking for any select elements...")
            all_selects = soup.find_all('select')
            print(f"Found {len(all_selects)} select elements total")
            
            for i, select in enumerate(all_selects):
                print(f"Select {i+1}: name='{select.get('name')}', id='{select.get('id')}', class='{select.get('class')}'")
                options = select.find_all('option')
                if len(options) > 1:  # More than just a placeholder
                    print(f"  Has {len(options)} options")
                    
                    # If this looks like it could be the route select
                    if 'route' in str(select.get('name', '')).lower() or 'route' in str(select.get('id', '')).lower():
                        print(f"  This might be the route select! Extracting options...")
                        for option in options:
                            value = option.get('value', '')
                            text = option.get_text(strip=True)
                            if value or text:
                                route_options.append({
                                    'value': value,
                                    'text': text
                                })
                                print(f"    Value: '{value}' | Text: '{text}'")
            
            if route_options:
                return route_options
            
            # If still no luck, save the HTML for manual inspection
            with open('page_content.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("Saved page content to 'page_content.html' for manual inspection")
            
            return []
            
        except requests.exceptions.SSLError as e:
            print(f"SSL Error with {approach['description']}: {e}")
            continue
        except requests.exceptions.RequestException as e:
            print(f"Request Error with {approach['description']}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected Error with {approach['description']}: {e}")
            continue
    
    print("All approaches failed.")
    return []

def scrape_with_custom_ssl_context():
    """
    Alternative approach using a custom SSL context
    """
    import ssl
    import urllib.request
    
    print("Trying with custom SSL context...")
    
    # Create unverified SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    url = "https://mtcbus.tn.gov.in/Home/routewiseinfo"
    
    try:
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        
        with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
            html_content = response.read().decode('utf-8')
            
        print(f"Success with urllib! Content length: {len(html_content)} bytes")
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Look for select elements
        selects = soup.find_all('select')
        print(f"Found {len(selects)} select elements")
        
        route_options = []
        for select in selects:
            if 'route' in str(select.get('name', '')).lower() or 'route' in str(select.get('id', '')).lower():
                options = select.find_all('option')
                print(f"Found route select with {len(options)} options")
                
                for option in options:
                    value = option.get('value', '')
                    text = option.get_text(strip=True)
                    if value or text:
                        route_options.append({
                            'value': value,
                            'text': text
                        })
                        print(f"Value: '{value}' | Text: '{text}'")
        
        return route_options
        
    except Exception as e:
        print(f"Custom SSL context approach failed: {e}")
        return []

def save_to_file(route_options, filename='route_options.json'):
    """
    Save the scraped route options to a JSON file
    """
    try:
        # Filter routes to only include meaningful ones
        filtered_routes = []
        
        for option in route_options:
            value = option['value']
            text = option['text']
            
            # Skip empty values or placeholder routes
            if not value or value == '':
                continue
                
            # Only include routes with meaningful text
            if is_meaningful_route_text(text):
                filtered_routes.append({
                    'route_id': value,
                    'route_name': text.strip()
                })
        
        # Create the output structure
        output_data = {
            'total_routes': len(filtered_routes),
            'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'routes': filtered_routes
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nFiltered route options saved to '{filename}'")
        return filtered_routes
        
    except Exception as e:
        print(f"Error saving to file: {e}")
        return []

def main():
    print("MTC Bus Route Scraper (JSON Output Version)")
    print("=" * 45)
    
    # Try the main approach first
    route_options = scrape_route_options()
    
    # If that fails, try the custom SSL context approach
    if not route_options:
        route_options = scrape_with_custom_ssl_context()
    
    if route_options:
        print(f"\nSuccessfully scraped {len(route_options)} route options!")
        
        # Save to JSON file and get filtered results
        filtered_routes = save_to_file(route_options)
        
        # Display summary
        print(f"\nSummary:")
        print(f"Total options found: {len(route_options)}")
        print(f"Meaningful routes (filtered): {len(filtered_routes)}")
        
        # Show first few filtered options as example
        print(f"\nFirst 10 meaningful routes:")
        for i, route in enumerate(filtered_routes[:10]):
            print(f"  {i+1}. ID: '{route['route_id']}' - Name: '{route['route_name']}'")
        
        if len(filtered_routes) > 10:
            print(f"  ... and {len(filtered_routes) - 10} more")
        
        # Also create a simple list for easy access
        simple_routes = [ r['route_name'] for r in filtered_routes]
        with open('simple_routes.json', 'w', encoding='utf-8') as f:
            json.dump(simple_routes, f, indent=2, ensure_ascii=False)
        print(f"\nSimple route list saved to 'simple_routes.json'")
            
    else:
        print("No route options found or error occurred during scraping.")
        print("\nTroubleshooting suggestions:")
        print("1. Check if 'page_content.html' was created and inspect it manually")
        print("2. The website might be temporarily down")
        print("3. The website structure might have changed")
        print("4. Try the Selenium version if this continues to fail")

if __name__ == "__main__":
    main()