import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from math import radians, cos, sin, asin, sqrt
import seaborn as sns

class DepotToSchoolMDVRP:
    def __init__(self, data_file):
        """Initialize MDVRP solver for depot-to-school routing"""
        self.load_data(data_file)
        self.create_distance_matrix()
        self.setup_problem_parameters()
        
    def load_data(self, data_file):
        """Load and analyze the stop data"""
        self.stops_df = pd.read_csv(data_file)
        
        print("="*60)
        print("MULTI-DEPOT VEHICLE ROUTING PROBLEM - DEPOT TO SCHOOL")
        print("="*60)
        
        # Analyze the data
        depots = self.stops_df[self.stops_df['is_depot'] == True]
        school = self.stops_df[self.stops_df['is_school'] == True]
        stops = self.stops_df[(self.stops_df['is_depot'] == False) & (self.stops_df['is_school'] == False)]
        
        print(f"\nData Analysis:")
        print(f"- Total locations: {len(self.stops_df)}")
        print(f"- Depots: {len(depots)}")
        print(f"- Regular stops: {len(stops)}")
        print(f"- School (destination): {len(school)}")
        print(f"- Total students to transport: {stops['num_students'].sum()}")
        
        print(f"\nDepots:")
        for _, depot in depots.iterrows():
            print(f"  {depot['stop_id']}: {depot['stop_name']} at ({depot['latitude']:.3f}, {depot['longitude']:.3f})")
            
        print(f"\nSchool (Destination):")
        for _, sch in school.iterrows():
            print(f"  {sch['stop_id']}: {sch['stop_name']} at ({sch['latitude']:.3f}, {sch['longitude']:.3f})")
            
        # Get indices for depots and school
        self.depot_indices = depots.index.tolist()
        self.school_index = school.index.tolist()[0]  # Single school
        self.stop_indices = stops.index.tolist()
        
        print(f"\nIndices:")
        print(f"- Depot indices: {self.depot_indices}")
        print(f"- School index: {self.school_index}")
        print(f"- Regular stop indices: {self.stop_indices}")
        
    def setup_problem_parameters(self):
        """Setup MDVRP parameters based on depot bus allocation"""
        # Bus allocation per depot (as specified)
        self.depot_bus_allocation = {
            0: 3,  # Depot A: 2 buses (index 0)
            1: 6,  # Depot B: 5 buses (index 1)
            2: 5,  # Depot C: 5 buses (index 2)
            3: 3   # Depot D: 3 buses (index 3)
        }
        
        self.vehicle_capacity = 55
        self.total_vehicles = sum(self.depot_bus_allocation.values())
        
        # Create vehicle start and end points
        self.vehicle_starts = []
        self.vehicle_ends = []
        
        vehicle_id = 0
        for depot_idx, num_buses in self.depot_bus_allocation.items():
            for _ in range(num_buses):
                self.vehicle_starts.append(depot_idx)  # Start at depot
                self.vehicle_ends.append(self.school_index)  # End at school
                vehicle_id += 1
        
        print(f"\nVehicle Configuration:")
        print(f"- Total vehicles: {self.total_vehicles}")
        print(f"- Vehicle capacity: {self.vehicle_capacity} students")
        print(f"- Vehicle starts (depot indices): {self.vehicle_starts}")
        print(f"- Vehicle ends (school index): {self.vehicle_ends}")
        
        total_students = self.stops_df[self.stops_df['is_depot'] == False]['num_students'].sum()
        total_capacity = self.total_vehicles * self.vehicle_capacity
        print(f"- Total students: {total_students}")
        print(f"- Total capacity: {total_capacity}")
        print(f"- Capacity utilization: {total_students/total_capacity*100:.1f}%")
        
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate road distance between two points"""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return int(6371000 * c * 1.3)  # Earth radius * road factor
        
    def create_distance_matrix(self):
        """Create distance matrix for all locations"""
        print(f"\nCreating distance matrix...")
        
        n_locations = len(self.stops_df)
        self.distance_matrix = np.zeros((n_locations, n_locations), dtype=int)
        
        for i in range(n_locations):
            for j in range(n_locations):
                if i == j:
                    self.distance_matrix[i][j] = 0
                else:
                    lat1, lon1 = self.stops_df.iloc[i]['latitude'], self.stops_df.iloc[i]['longitude']
                    lat2, lon2 = self.stops_df.iloc[j]['latitude'], self.stops_df.iloc[j]['longitude']
                    self.distance_matrix[i][j] = self.haversine_distance(lat1, lon1, lat2, lon2)
        
        print(f"Distance matrix created: {self.distance_matrix.shape}")
        
        # Save distance matrix for reference
        distance_df = pd.DataFrame(
            self.distance_matrix,
            index=self.stops_df['stop_id'],
            columns=self.stops_df['stop_id']
        )
        distance_df.to_csv('distance_matrix_depot_school.csv')
        print("Distance matrix saved to 'distance_matrix_depot_school.csv'")
        
    def solve_mdvrp(self, time_limit_seconds=300):
        """Solve the MDVRP problem"""
        print(f"\n{'='*60}")
        print("SOLVING MDVRP...")
        print(f"{'='*60}")
        
        # Prepare demands (students at each stop)
        demands = self.stops_df['num_students'].tolist()
        
        # Create data model for OR-Tools
        data = {
            'distance_matrix': self.distance_matrix,
            'demands': demands,
            'vehicle_capacities': [self.vehicle_capacity] * self.total_vehicles,
            'num_vehicles': self.total_vehicles,
            'starts': self.vehicle_starts,
            'ends': self.vehicle_ends
        }
        
        print(f"Problem setup complete:")
        print(f"- Vehicles: {data['num_vehicles']}")
        print(f"- Starts: {data['starts']}")
        print(f"- Ends: {data['ends']}")
        print(f"- Total demand: {sum(demands)} students")
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']),
            data['num_vehicles'],
            data['starts'],
            data['ends']
        )
        
        routing = pywrapcp.RoutingModel(manager)
        
        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Capacity constraint
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],
            True,  # start cumul to zero
            'Capacity'
        )
        
        # Search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(time_limit_seconds)
        
        print(f"Solving with time limit: {time_limit_seconds} seconds...")
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self.process_solution(data, manager, routing, solution)
        else:
            print("No solution found!")
            return None
            
    def process_solution(self, data, manager, routing, solution):
        """Process and display the solution"""
        print(f"\n{'='*50}")
        print("SOLUTION FOUND!")
        print(f"{'='*50}")
        print(f"Objective (Total Distance): {solution.ObjectiveValue():,} meters")
        print(f"Objective (Total Distance): {solution.ObjectiveValue()/1000:.1f} km")
        
        routes = []
        total_distance = 0
        total_students = 0
        depot_usage = {i: 0 for i in range(len(self.depot_indices))}
        
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route_distance = 0
            route_students = 0
            route_stops = []
            
            # Build route
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_stops.append(node_index)
                route_students += data['demands'][node_index]
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            
            # Add final destination (school)
            final_node = manager.IndexToNode(index)
            route_stops.append(final_node)
            
            if len(route_stops) > 2:  # Routes with actual stops
                start_depot_idx = route_stops[0]
                depot_usage[start_depot_idx] += 1
                
                route_info = {
                    'vehicle_id': vehicle_id,
                    'stops': route_stops,
                    'distance_m': route_distance,
                    'distance_km': route_distance / 1000,
                    'students': route_students,
                    'capacity_utilization': (route_students / self.vehicle_capacity) * 100,
                    'start_depot': start_depot_idx,
                    'start_depot_name': self.stops_df.iloc[start_depot_idx]['stop_name'],
                    'num_pickup_stops': len(route_stops) - 2  # Exclude depot and school
                }
                routes.append(route_info)
                
                total_distance += route_distance
                total_students += route_students
                
                print(f"\n--- Bus {vehicle_id} ---")
                print(f"Start Depot: {route_info['start_depot_name']}")
                print(f"Route: {' → '.join([self.stops_df.iloc[stop]['stop_name'] for stop in route_stops])}")
                print(f"Pickup Stops: {route_info['num_pickup_stops']}")
                print(f"Distance: {route_distance:,} meters ({route_distance/1000:.1f} km)")
                print(f"Students: {route_students}/{self.vehicle_capacity} ({route_students/self.vehicle_capacity*100:.1f}% capacity)")
            else:
                # Empty route
                start_depot_idx = route_stops[0]
                print(f"\n--- Bus {vehicle_id} (EMPTY ROUTE) ---")
                print(f"Start Depot: {self.stops_df.iloc[start_depot_idx]['stop_name']}")
                print(f"Route: {self.stops_df.iloc[start_depot_idx]['stop_name']} → {self.stops_df.iloc[route_stops[-1]]['stop_name']}")
                print(f"Distance: {route_distance:,} meters ({route_distance/1000:.1f} km)")
                print(f"Students: 0/{self.vehicle_capacity} (0% capacity)")
        
        # Summary statistics
        print(f"\n{'='*50}")
        print("SOLUTION SUMMARY")
        print(f"{'='*50}")
        print(f"Total Distance: {total_distance:,} meters ({total_distance/1000:.1f} km)")
        print(f"Total Students Transported: {total_students}")
        print(f"Active Vehicles: {len(routes)}/{self.total_vehicles}")
        if len(routes) > 0:
            print(f"Average Distance per Active Vehicle: {total_distance/len(routes):,.0f} meters")
            print(f"Average Students per Active Vehicle: {total_students/len(routes):.1f}")
            print(f"Overall Capacity Utilization: {total_students/(len(routes)*self.vehicle_capacity)*100:.1f}%")
        
        print(f"\nDepot Usage:")
        for depot_idx, count in depot_usage.items():
            depot_name = self.stops_df.iloc[depot_idx]['stop_name']
            allocated = self.depot_bus_allocation[depot_idx]
            print(f"- {depot_name}: {count}/{allocated} buses used")
        
        solution_data = {
            'routes': routes,
            'total_distance': total_distance,
            'total_students': total_students,
            'vehicles_used': len(routes),
            'depot_usage': depot_usage,
            'objective_value': solution.ObjectiveValue()
        }
        
        # Save detailed solution
        self.save_detailed_solution(routes)
        
        return solution_data
        
    def save_detailed_solution(self, routes):
        """Save detailed solution to CSV"""
        detailed_data = []
        
        for route in routes:
            for seq, stop_id in enumerate(route['stops']):
                stop_info = self.stops_df.iloc[stop_id]
                detailed_data.append({
                    'bus_id': route['vehicle_id'],
                    'sequence': seq,
                    'stop_id': stop_info['stop_id'],
                    'stop_name': stop_info['stop_name'],
                    'latitude': stop_info['latitude'],
                    'longitude': stop_info['longitude'],
                    'students_pickup': stop_info['num_students'],
                    'is_depot': stop_info['is_depot'],
                    'is_school': stop_info['is_school'],
                    'route_total_distance_m': route['distance_m'],
                    'route_total_distance_km': route['distance_km'],
                    'route_total_students': route['students'],
                    'route_capacity_utilization': route['capacity_utilization'],
                    'start_depot': route['start_depot_name']
                })
        
        solution_df = pd.DataFrame(detailed_data)
        solution_df.to_csv('mdvrp_depot_school_solution.csv', index=False)
        print(f"\nDetailed solution saved to 'mdvrp_depot_school_solution.csv'")
        
    def visualize_solution(self, solution_data):
        """Create comprehensive visualization of the solution"""
        print(f"\nCreating visualizations...")
        
        # Main route visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot 1: Route Map
        colors = plt.cm.Set1(np.linspace(0, 1, max(8, len(solution_data['routes']))))
        
        # Plot all locations
        for idx, row in self.stops_df.iterrows():
            if row['is_depot']:
                ax1.scatter(row['longitude'], row['latitude'], 
                           c='red', s=400, marker='s', 
                           edgecolors='black', linewidth=2,
                           label='Depot' if idx == 0 else "", zorder=5)
            elif row['is_school']:
                ax1.scatter(row['longitude'], row['latitude'], 
                           c='blue', s=400, marker='^', 
                           edgecolors='black', linewidth=2,
                           label='School', zorder=5)
            else:
                ax1.scatter(row['longitude'], row['latitude'], 
                           c='lightgray', s=100, marker='o',
                           edgecolors='black', linewidth=1,
                           alpha=0.7, zorder=3)
        
        # Plot routes
        for i, route in enumerate(solution_data['routes']):
            route_lons = [self.stops_df.iloc[stop]['longitude'] for stop in route['stops']]
            route_lats = [self.stops_df.iloc[stop]['latitude'] for stop in route['stops']]
            
            # Plot route line
            ax1.plot(route_lons, route_lats, 
                    color=colors[i], linewidth=3, alpha=0.8,
                    label=f"Bus {route['vehicle_id']} ({route['students']}👥, {route['distance_km']:.1f}km)", zorder=4)
            
            # Plot pickup stops for this route
            for j, stop in enumerate(route['stops'][1:-1], 1):  # Exclude depot and school
                ax1.scatter(self.stops_df.iloc[stop]['longitude'], 
                           self.stops_df.iloc[stop]['latitude'],
                           c=[colors[i]], s=150, marker='o',
                           edgecolors='black', linewidth=1, zorder=4)
                
                # Add sequence numbers
                ax1.annotate(f"{j}", 
                           (self.stops_df.iloc[stop]['longitude'], 
                            self.stops_df.iloc[stop]['latitude']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold', color='white',
                           bbox=dict(boxstyle="circle,pad=0.1", facecolor=colors[i], alpha=0.8))
        
        # Add location labels
        for idx, row in self.stops_df.iterrows():
            if row['is_depot']:
                ax1.annotate(f"{row['stop_name']}", 
                           (row['longitude'], row['latitude']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            elif row['is_school']:
                ax1.annotate(f"{row['stop_name']}", 
                           (row['longitude'], row['latitude']),
                           xytext=(10, -20), textcoords='offset points',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            elif not row['is_depot'] and not row['is_school']:
                ax1.annotate(f"{row['stop_name']}\n({row['num_students']}👥)", 
                           (row['longitude'], row['latitude']),
                           xytext=(5, -15), textcoords='offset points',
                           fontsize=7, ha='center',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        
        ax1.set_xlabel('Longitude', fontsize=12)
        ax1.set_ylabel('Latitude', fontsize=12)
        ax1.set_title(f'MDVRP Routes: Depots → School\n'
                     f'Total: {solution_data["total_distance"]/1000:.1f}km, '
                     f'{solution_data["total_students"]} students, '
                     f'{solution_data["vehicles_used"]} buses', 
                     fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Statistics
        if solution_data['routes']:
            # Bus utilization
            buses = [f"Bus {r['vehicle_id']}" for r in solution_data['routes']]
            students = [r['students'] for r in solution_data['routes']]
            distances = [r['distance_km'] for r in solution_data['routes']]
            
            x = np.arange(len(buses))
            width = 0.35
            
            ax2_twin = ax2.twinx()
            
            bars1 = ax2.bar(x - width/2, students, width, label='Students', color='lightgreen', alpha=0.8)
            bars2 = ax2_twin.bar(x + width/2, distances, width, label='Distance (km)', color='orange', alpha=0.8)
            
            # Add capacity line
            ax2.axhline(y=self.vehicle_capacity, color='red', linestyle='--', linewidth=2, label=f'Capacity ({self.vehicle_capacity})')
            
            ax2.set_xlabel('Buses')
            ax2.set_ylabel('Students', color='green')
            ax2_twin.set_ylabel('Distance (km)', color='orange')
            ax2.set_title('Bus Utilization: Students & Distance')
            ax2.set_xticks(x)
            ax2.set_xticklabels(buses, rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars1, students):
                ax2.annotate(f'{value}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=8)
            
            for bar, value in zip(bars2, distances):
                ax2_twin.annotate(f'{value:.1f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                               ha='center', va='bottom', fontsize=8)
            
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mdvrp_depot_school_visualization.png', dpi=300, bbox_inches='tight')
        print("Main visualization saved as 'mdvrp_depot_school_visualization.png'")
        plt.show()
        
        # Additional depot usage chart
        self.create_depot_usage_chart(solution_data)
        
    def create_depot_usage_chart(self, solution_data):
        """Create depot usage visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Depot usage
        depot_names = [self.stops_df.iloc[idx]['stop_name'] for idx in self.depot_indices]
        allocated = [self.depot_bus_allocation[idx] for idx in range(len(self.depot_indices))]
        used = [solution_data['depot_usage'][idx] for idx in range(len(self.depot_indices))]
        
        x = np.arange(len(depot_names))
        width = 0.35
        
        ax1.bar(x - width/2, allocated, width, label='Allocated', color='lightblue', alpha=0.8)
        ax1.bar(x + width/2, used, width, label='Used', color='darkblue', alpha=0.8)
        
        ax1.set_xlabel('Depots')
        ax1.set_ylabel('Number of Buses')
        ax1.set_title('Bus Allocation vs Usage by Depot')
        ax1.set_xticks(x)
        ax1.set_xticklabels(depot_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (alloc, use) in enumerate(zip(allocated, used)):
            ax1.text(i - width/2, alloc + 0.05, str(alloc), ha='center', va='bottom')
            ax1.text(i + width/2, use + 0.05, str(use), ha='center', va='bottom')
        
        # Distance distribution pie chart
        if solution_data['routes']:
            route_distances = [r['distance_km'] for r in solution_data['routes']]
            route_labels = [f"Bus {r['vehicle_id']}\n({r['distance_km']:.1f}km)" for r in solution_data['routes']]
            
            ax2.pie(route_distances, labels=route_labels, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Distance Distribution by Bus')
        
        plt.tight_layout()
        plt.savefig('mdvrp_depot_usage_stats.png', dpi=300, bbox_inches='tight')
        print("Depot usage chart saved as 'mdvrp_depot_usage_stats.png'")
        plt.show()

def main():
    """Run the depot-to-school MDVRP solver"""
    print("Loading MDVRP data...")
    
    # Create solver
    solver = DepotToSchoolMDVRP('mdvrp_data.csv')
    
    # Solve the problem
    solution = solver.solve_mdvrp(time_limit_seconds=300)
    
    if solution:
        # Visualize results
        solver.visualize_solution(solution)
        
        print(f"\n{'='*60}")
        print("MDVRP SOLUTION COMPLETED!")
        print(f"{'='*60}")
        print("Generated files:")
        print("1. distance_matrix_depot_school.csv - Distance matrix")
        print("2. mdvrp_depot_school_solution.csv - Detailed route solution")
        print("3. mdvrp_depot_school_visualization.png - Route map & statistics")
        print("4. mdvrp_depot_usage_stats.png - Depot usage charts")
        
        # Final insights
        print(f"\nKey Results:")
        print(f"✓ Successfully routed {solution['total_students']} students")
        print(f"✓ Used {solution['vehicles_used']}/{solver.total_vehicles} buses")
        print(f"✓ Total travel distance: {solution['total_distance']/1000:.1f} km")
        print(f"✓ Average distance per bus: {solution['total_distance']/(solution['vehicles_used']*1000):.1f} km")
        print(f"✓ All buses end at the school as required")
        print(f"✓ All capacity constraints satisfied")
        
        # Check if all students are covered
        total_students_in_data = solver.stops_df[solver.stops_df['is_depot'] == False]['num_students'].sum()
        if solution['total_students'] == total_students_in_data:
            print(f"✓ All {total_students_in_data} students successfully assigned to routes")
        else:
            print(f"⚠ {total_students_in_data - solution['total_students']} students not assigned!")
            
    else:
        print("❌ Failed to find a solution!")
        print("Try increasing time limit or adjusting constraints.")

if __name__ == "__main__":
    main()