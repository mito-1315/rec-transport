#!/usr/bin/env python3
"""
MTC Route Data Analyzer
Analyze and process scraped MTC bus route data
"""

import json
import csv
from collections import Counter, defaultdict
import sys
from typing import Dict, List

class RouteAnalyzer:
    def __init__(self, data_file: str):
        """Initialize with scraped route data."""
        self.data = self.load_data(data_file)
        self.routes = self.data.get('routes', {})
    
    def load_data(self, filename: str) -> Dict:
        """Load scraped route data from JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: {filename} not found!")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing {filename}: {e}")
            return {}
    
    def get_statistics(self) -> Dict:
        """Generate statistics about the routes."""
        if not self.routes:
            return {}
        
        total_routes = len(self.routes)
        total_stops = sum(len(route_data['stops']) for route_data in self.routes.values())
        
        # Stop counts per route
        stop_counts = [len(route_data['stops']) for route_data in self.routes.values()]
        
        # Most common stop names
        all_stop_names = []
        for route_data in self.routes.values():
            all_stop_names.extend([stop['stop_name'] for stop in route_data['stops']])
        
        common_stops = Counter(all_stop_names).most_common(20)
        
        stats = {
            'total_routes': total_routes,
            'total_stops': total_stops,
            'average_stops_per_route': total_stops / total_routes if total_routes > 0 else 0,
            'min_stops': min(stop_counts) if stop_counts else 0,
            'max_stops': max(stop_counts) if stop_counts else 0,
            'most_common_stops': common_stops,
            'routes_by_stop_count': Counter(stop_counts)
        }
        
        return stats
    
    def find_routes_by_stop(self, stop_name: str) -> List[str]:
        """Find all routes that pass through a specific stop."""
        matching_routes = []
        
        for route_num, route_data in self.routes.items():
            for stop in route_data['stops']:
                if stop_name.lower() in stop['stop_name'].lower():
                    matching_routes.append(route_num)
                    break
        
        return matching_routes
    
    def find_common_routes(self, stop1: str, stop2: str) -> List[str]:
        """Find routes that connect two stops."""
        routes_with_stop1 = set(self.find_routes_by_stop(stop1))
        routes_with_stop2 = set(self.find_routes_by_stop(stop2))
        
        return list(routes_with_stop1.intersection(routes_with_stop2))
    
    def export_route_summary(self, filename: str = 'route_summary.csv'):
        """Export a summary of all routes to CSV."""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['route_number', 'total_stops', 'first_stop', 'last_stop']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for route_num, route_data in self.routes.items():
                    stops = route_data['stops']
                    writer.writerow({
                        'route_number': route_num,
                        'total_stops': len(stops),
                        'first_stop': stops[0]['stop_name'] if stops else '',
                        'last_stop': stops[-1]['stop_name'] if stops else ''
                    })
            
            print(f"Route summary exported to {filename}")
        except Exception as e:
            print(f"Error exporting route summary: {e}")
    
    def print_statistics(self):
        """Print route statistics."""
        stats = self.get_statistics()
        
        if not stats:
            print("No data available for analysis.")
            return
        
        print("MTC Bus Routes Statistics")
        print("=" * 40)
        print(f"Total Routes: {stats['total_routes']}")
        print(f"Total Stops: {stats['total_stops']}")
        print(f"Average Stops per Route: {stats['average_stops_per_route']:.1f}")
        print(f"Min Stops in a Route: {stats['min_stops']}")
        print(f"Max Stops in a Route: {stats['max_stops']}")
        
        print(f"\nMost Common Stops:")
        for i, (stop_name, count) in enumerate(stats['most_common_stops'][:10], 1):
            print(f"{i:2d}. {stop_name} ({count} routes)")
        
        print(f"\nRoutes by Stop Count:")
        for stop_count, route_count in sorted(stats['routes_by_stop_count'].items())[:10]:
            print(f"{route_count} routes have {stop_count} stops")

def main():
    if len(sys.argv) < 2:
        print("Usage: python route_analyzer.py <scraped_data_file.json>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    analyzer = RouteAnalyzer(data_file)
    
    if not analyzer.routes:
        print("No route data found.")
        sys.exit(1)
    
    # Print statistics
    analyzer.print_statistics()
    
    # Export summary
    analyzer.export_route_summary()
    
    # Interactive mode
    print(f"\n" + "="*40)
    print("Interactive Mode")
    print("="*40)
    
    while True:
        print(f"\nOptions:")
        print("1. Find routes by stop name")
        print("2. Find common routes between two stops")
        print("3. Show route details")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            stop_name = input("Enter stop name (partial match): ").strip()
            if stop_name:
                routes = analyzer.find_routes_by_stop(stop_name)
                print(f"\nRoutes passing through '{stop_name}': {', '.join(routes) if routes else 'None found'}")
        
        elif choice == '2':
            stop1 = input("Enter first stop name: ").strip()
            stop2 = input("Enter second stop name: ").strip()
            if stop1 and stop2:
                routes = analyzer.find_common_routes(stop1, stop2)
                print(f"\nRoutes connecting '{stop1}' and '{stop2}': {', '.join(routes) if routes else 'None found'}")
        
        elif choice == '3':
            route_num = input("Enter route number: ").strip()
            if route_num in analyzer.routes:
                route_data = analyzer.routes[route_num]
                print(f"\nRoute {route_num} Details:")
                print(f"Total Stops: {len(route_data['stops'])}")
                print("Stops:")
                for stop in route_data['stops']:
                    print(f"  {stop['stop_number']}. {stop['stop_name']}")
            else:
                print(f"Route {route_num} not found.")
        
        elif choice == '4':
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()