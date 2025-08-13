import os
from dotenv import load_dotenv
from route_optimization import EnhancedBusRouteOptimizer
x
def main():
    """
    Main function demonstrating the enhanced bus route optimizer with individual route filtering
    """
    # Initialize optimizer
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_MAP_API_KEY")
    
    if not API_KEY:
        print("Warning: No Google Maps API key found. Using basic functionality only.")
        API_KEY = "dummy_key"
    
    optimizer = EnhancedBusRouteOptimizer(API_KEY)
    
    # File paths
    bus_stops_file = "Routes_Data/Friday/3_pm/3_pm_centroids_snapped.csv"
    depots_file = "Routes_Data/Bus_Stops/coords.csv"
    college_coords = (13.008794724595475, 80.00342657961114)
    
    try:
        print("🚌 Starting Enhanced Bus Route Optimization with Individual Route Filtering...")
        print("-" * 80)
        
        # Load data
        print("📁 Loading data...")
        optimizer.load_data(bus_stops_file, depots_file, college_coords)
        
        # Optimize routes
        print("\n🔄 Optimizing routes...")
        routes = optimizer.optimize_routes()
        
        if routes:
            # Analyze results
            print("\n📊 Analyzing results...")
            optimizer.analyze_results()
            
            # Export comprehensive CSV files
            print("\n💾 Exporting data to CSV files...")
            exported_files = optimizer.export_comprehensive_csvs()
            
            # Create advanced interactive map with individual route filtering
            print("\n🗺️ Creating advanced interactive map with individual route filtering...")
            optimizer.create_advanced_interactive_map()
            
            # Create individual route maps
            print("\n🎯 Creating detailed individual route maps...")
            individual_maps_dir = optimizer.create_individual_route_maps()
            
            # Demonstrate viewing a specific route
            print("\n🔍 Viewing individual route details...")
            route_dir = optimizer.view_individual_route()  # Will show menu to select route
            
            print("\n" + "="*70)
            print("✅ OPTIMIZATION COMPLETE WITH ENHANCED FEATURES!")
            print("="*70)
            print("📁 Generated Files:")
            print(f"   • Enhanced interactive map: {optimizer.output_dir}/interactive_bus_routes_*.html")
            print(f"     Features: Individual route filtering, efficiency filters, comparison dashboard")
            print(f"   • Individual route maps: {individual_maps_dir}")
            print(f"     Features: Detailed stop analysis, capacity visualization, route statistics")
            print("   • CSV exports:")
            for file_type, file_path in exported_files.items():
                print(f"     - {file_type.replace('_', ' ').title()}: {file_path}")
            
            print(f"\n💡 Enhanced Features Available:")
            print(f"   • 🎛️ Individual route toggle buttons for precise filtering")
            print(f"   • 🔍 Smart filters (efficiency, distance-based)")
            print(f"   • 📊 Interactive route comparison dashboard")
            print(f"   • 📈 Visual capacity utilization indicators")
            print(f"   • 🗺️ Minimizable control panel for better map viewing")
            print(f"   • 🎯 Detailed individual route maps with enhanced popups")
            print(f"   • 📋 Comprehensive CSV exports for data analysis")
            print(f"   • 🔎 View details of any specific route using view_individual_route() method")
            print("="*70)
            
        else:
            print("❌ Route optimization failed.")
            
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please ensure your CSV files are in the correct location")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()