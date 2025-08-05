#!/usr/bin/env python3
"""
Simple runner script for MTC route scraper
"""

import subprocess
import sys
import os

def main():
    print("MTC Bus Route Scraper Runner")
    print("=" * 40)
    
    # Check if simple_routes.json exists
    if not os.path.exists('simple_routes.json'):
        print("❌ Error: simple_routes.json not found!")
        print("Please make sure the file is in the current directory.")
        sys.exit(1)
    
    print("✓ Found simple_routes.json")
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Scrape all routes (full scraping)")
    print("2. Scrape specific route numbers")
    print("3. Resume interrupted scraping")
    print("4. Analyze existing scraped data")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    try:
        if choice == '1':
            print("\n🚀 Starting full route scraping...")
            subprocess.run([sys.executable, 'scrape_all_routes.py'], check=True)
        
        elif choice == '2':
            route_numbers = input("Enter route numbers (comma-separated): ").strip()
            if route_numbers:
                # Create a temporary JSON file with specified routes
                import json
                routes = [r.strip() for r in route_numbers.split(',')]
                with open('temp_routes.json', 'w') as f:
                    json.dump(routes, f)
                print(f"\n🚀 Scraping {len(routes)} specific routes...")
                # Modify the scraper to use temp_routes.json
                subprocess.run([sys.executable, 'scrape_all_routes.py'], check=True)
                os.remove('temp_routes.json')
        
        elif choice == '3':
            print("\n🔄 Resuming interrupted scraping...")
            # The main scraper handles resume functionality
            subprocess.run([sys.executable, 'scrape_all_routes.py'], check=True)
        
        elif choice == '4':
            # Find the most recent scraped data file
            import glob
            json_files = glob.glob('mtc_all_routes_*.json')
            if json_files:
                latest_file = max(json_files, key=os.path.getctime)
                print(f"\n📊 Analyzing {latest_file}...")
                subprocess.run([sys.executable, 'route_analyzer.py', latest_file], check=True)
            else:
                print("❌ No scraped data files found.")
        
        else:
            print("Invalid choice.")
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running script: {e}")
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user.")

if __name__ == "__main__":
    main()