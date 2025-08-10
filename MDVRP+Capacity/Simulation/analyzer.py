import pandas as pd
import numpy as np

def analyze_solution():
    """Analyze the MDVRP solution results"""
    
    # Load the solution data
    try:
        solution_df = pd.read_csv('mdvrp_depot_school_solution.csv')
        print("="*60)
        print("MDVRP SOLUTION ANALYSIS")
        print("="*60)
        
        # Basic statistics
        total_buses = solution_df['bus_id'].nunique()
        total_students = solution_df['students_pickup'].sum()
        total_distance = solution_df['route_total_distance_km'].iloc[0] * total_buses if len(solution_df) > 0 else 0
        
        print(f"\nOverall Statistics:")
        print(f"- Total buses used: {total_buses}")
        print(f"- Total students transported: {total_students}")
        print(f"- Total distance: {solution_df.groupby('bus_id')['route_total_distance_km'].first().sum():.1f} km")
        
        # Bus-wise analysis
        print(f"\nBus-wise Analysis:")
        bus_summary = solution_df.groupby('bus_id').agg({
            'students_pickup': 'sum',
            'route_total_distance_km': 'first',
            'route_capacity_utilization': 'first',
            'start_depot': 'first'
        }).round(1)
        
        bus_summary['stops_count'] = solution_df.groupby('bus_id').size() - 2  # Exclude depot and school
        
        print(bus_summary.to_string())
        
        # Depot-wise analysis
        print(f"\nDepot-wise Analysis:")
        depot_summary = solution_df.groupby('start_depot').agg({
            'bus_id': 'nunique',
            'students_pickup': 'sum',
            'route_total_distance_km': 'sum'
        }).round(1)
        depot_summary.columns = ['Buses_Used', 'Total_Students', 'Total_Distance_km']
        
        print(depot_summary.to_string())
        
        # Route details
        print(f"\nDetailed Routes:")
        for bus_id in sorted(solution_df['bus_id'].unique()):
            bus_route = solution_df[solution_df['bus_id'] == bus_id].sort_values('sequence')
            route_names = ' → '.join(bus_route['stop_name'].tolist())
            students = bus_route['students_pickup'].sum()
            distance = bus_route['route_total_distance_km'].iloc[0]
            depot = bus_route['start_depot'].iloc[0]
            
            print(f"\nBus {bus_id} (from {depot}):")
            print(f"  Route: {route_names}")
            print(f"  Students: {students}/55 ({students/55*100:.1f}% capacity)")
            print(f"  Distance: {distance:.1f} km")
            print(f"  Stops: {len(bus_route) - 2} pickup stops")
        
    except FileNotFoundError:
        print("Solution file not found. Please run the MDVRP solver first.")

if __name__ == "__main__":
    analyze_solution()