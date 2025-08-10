import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import matplotlib.pyplot as plt
import json

class MDVRPSolver:
    def __init__(self, distance_matrix_file, stops_data_file):
        """
        Initialize MDVRP solver
        
        Parameters:
        - distance_matrix_file: CSV file with distance matrix
        - stops_data_file: CSV file with stop information (students count, coordinates, etc.)
        """
        self.distance_matrix = self.load_distance_matrix(distance_matrix_file)
        self.stops_data = pd.read_csv(stops_data_file)
        self.num_locations = len(self.stops_data)
        
        # Default parameters (can be modified)
        self.num_vehicles = 130  # Number of buses
        self.vehicle_capacity = 55  # Default capacity per bus
        self.depot_indices = [0]  # Default depot is first location
        
        print(f"Loaded {self.num_locations} locations")
        print(f"Distance matrix shape: {self.distance_matrix.shape}")
    
    def load_distance_matrix(self, file_path):
        """Load distance matrix from CSV file"""
        df = pd.read_csv(file_path, index_col=0)
        return df.values.astype(int)
    
    def set_vehicle_parameters(self, num_vehicles, vehicle_capacity):
        """Set number of vehicles and their capacity"""
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        print(f"Set {num_vehicles} vehicles with capacity {vehicle_capacity} students each")
    
    def set_depots(self, depot_indices):
        """Set depot locations (can be multiple depots)"""
        self.depot_indices = depot_indices
        print(f"Set depots at indices: {depot_indices}")
    
    def get_demands(self):
        """Get student demands for each location"""
        if 'num_students' in self.stops_data.columns:
            demands = self.stops_data['num_students'].tolist()
        else:
            # Default to 1 student per stop if no data available
            demands = [1] * self.num_locations
        
        # Depots have 0 demand
        for depot_idx in self.depot_indices:
            demands[depot_idx] = 0
            
        return demands
    
    def create_data_model(self):
        """Create data model for the MDVRP"""
        data = {}
        data['distance_matrix'] = self.distance_matrix
        data['demands'] = self.get_demands()
        data['vehicle_capacities'] = [self.vehicle_capacity] * self.num_vehicles
        data['num_vehicles'] = self.num_vehicles
        data['starts'] = self.depot_indices * self.num_vehicles  # All vehicles start from depots
        data['ends'] = self.depot_indices * self.num_vehicles    # All vehicles end at depots
        
        return data
    
    def solve(self, time_limit_seconds=300):
        """
        Solve the MDVRP
        
        Parameters:
        - time_limit_seconds: Maximum time to spend solving
        """
        print("Creating optimization model...")
        
        # Create the routing index manager
        data = self.create_data_model()
        
        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']),
            data['num_vehicles'],
            data['starts'],
            data['ends']
        )
        
        # Create routing model
        routing = pywrapcp.RoutingModel(manager)
        
        # Define distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add capacity constraint
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity'
        )
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(time_limit_seconds)
        
        print(f"Solving MDVRP with {self.num_vehicles} vehicles...")
        print(f"Time limit: {time_limit_seconds} seconds")
        
        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self.process_solution(data, manager, routing, solution)
        else:
            print("No solution found!")
            return None
    
    def process_solution(self, data, manager, routing, solution):
        """Process and return the solution"""
        print(f"Solution found!")
        print(f"Objective: {solution.ObjectiveValue()} meters")
        
        total_distance = 0
        total_students = 0
        routes = []
        
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route_distance = 0
            route_students = 0
            route_stops = []
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_stops.append(node_index)
                route_students += data['demands'][node_index]
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            
            # Add final node (depot)
            final_node = manager.IndexToNode(index)
            route_stops.append(final_node)
            
            if len(route_stops) > 2:  # Only count routes with actual stops (not just depot->depot)
                routes.append({
                    'vehicle_id': vehicle_id,
                    'stops': route_stops,
                    'distance': route_distance,
                    'students': route_students,
                    'stop_details': [self.get_stop_info(stop) for stop in route_stops]
                })
                
                total_distance += route_distance
                total_students += route_students
                
                print(f"\nVehicle {vehicle_id}:")
                print(f"  Route: {' -> '.join(map(str, route_stops))}")
                print(f"  Distance: {route_distance} meters")
                print(f"  Students: {route_students}/{self.vehicle_capacity}")
        
        solution_summary = {
            'routes': routes,
            'total_distance': total_distance,
            'total_students': total_students,
            'num_vehicles_used': len(routes),
            'objective_value': solution.ObjectiveValue()
        }
        
        print(f"\n=== SOLUTION SUMMARY ===")
        print(f"Total distance: {total_distance:,} meters ({total_distance/1000:.1f} km)")
        print(f"Total students served: {total_students}")
        print(f"Vehicles used: {len(routes)}/{self.num_vehicles}")
        print(f"Average distance per vehicle: {total_distance/len(routes):,.0f} meters")
        
        return solution_summary
    
    def get_stop_info(self, stop_index):
        """Get detailed information about a stop"""
        if stop_index < len(self.stops_data):
            stop_data = self.stops_data.iloc[stop_index]
            return {
                'index': stop_index,
                'cluster_number': int(stop_data.get('cluster_number', stop_index)),
                'latitude': float(stop_data.get('latitude', 0)),
                'longitude': float(stop_data.get('longitude', 0)),
                'students': int(stop_data.get('num_students', 0)),
                'route_name': str(stop_data.get('route_name', f'Stop {stop_index}'))
            }
        return {'index': stop_index, 'students': 0}
    
    def save_solution(self, solution, filename):
        """Save solution to JSON file"""
        if solution:
            with open(filename, 'w') as f:
                json.dump(solution, f, indent=2, default=str)
            print(f"Solution saved to {filename}")
    
    def visualize_routes(self, solution, save_plot=True):
        """Visualize the routes on a map"""
        if not solution:
            print("No solution to visualize")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot all stops
        lats = self.stops_data['latitude'].values
        lons = self.stops_data['longitude'].values
        
        # Plot depot(s)
        for depot_idx in self.depot_indices:
            plt.scatter(lons[depot_idx], lats[depot_idx], 
                       c='red', s=200, marker='s', 
                       label='Depot' if depot_idx == self.depot_indices[0] else "")
        
        # Define colors for different routes
        colors = plt.cm.Set3(np.linspace(0, 1, len(solution['routes'])))
        
        for i, route in enumerate(solution['routes']):
            route_lons = [lons[stop] for stop in route['stops']]
            route_lats = [lats[stop] for stop in route['stops']]
            
            # Plot route line
            plt.plot(route_lons, route_lats, color=colors[i], linewidth=2, alpha=0.7,
                    label=f"Vehicle {route['vehicle_id']} ({route['students']} students)")
            
            # Plot stops
            plt.scatter(route_lons[1:-1], route_lats[1:-1], 
                       c=[colors[i]], s=50, alpha=0.8)
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'MDVRP Solution - {len(solution["routes"])} Routes')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('../output/mdvrp_routes.png', dpi=300, bbox_inches='tight')
            print("Route visualization saved as 'mdvrp_routes.png'")
        
        plt.show()

def main():
    # Initialize solver
    solver = MDVRPSolver(
        distance_matrix_file='../output/distance_matrix_road.csv',
        stops_data_file='../output/snapped.csv'
    )
    
    # Configure vehicles and constraints
    print("=== MDVRP Configuration ===")
    
    # Get user input for configuration
    num_vehicles = int(input("Enter number of buses (default 10): ") or "10")
    vehicle_capacity = int(input("Enter bus capacity in students (default 50): ") or "50")
    
    # Set depot(s) - you can modify this to use multiple depots
    print("Available stops for depot selection:")
    print(solver.stops_data[['cluster_number', 'route_name', 'num_students']].head(10))
    depot_input = input("Enter depot cluster number (default 0): ") or "0"
    depot_cluster = int(depot_input)
    
    # Find depot index
    depot_idx = solver.stops_data[solver.stops_data['cluster_number'] == depot_cluster].index[0]
    
    solver.set_vehicle_parameters(num_vehicles, vehicle_capacity)
    solver.set_depots([depot_idx])
    
    # Show problem statistics
    demands = solver.get_demands()
    total_students = sum(demands)
    print(f"\n=== Problem Statistics ===")
    print(f"Total students to transport: {total_students}")
    print(f"Total vehicle capacity: {num_vehicles * vehicle_capacity}")
    print(f"Capacity utilization: {total_students/(num_vehicles * vehicle_capacity)*100:.1f}%")
    print(f"Average students per stop: {total_students/len([d for d in demands if d > 0]):.1f}")
    
    # Solve the problem
    time_limit = int(input("Enter time limit in seconds (default 300): ") or "300")
    solution = solver.solve(time_limit_seconds=time_limit)
    
    if solution:
        # Save solution
        solver.save_solution(solution, '../output/mdvrp_solution.json')
        
        # Create detailed CSV output
        create_detailed_output(solution, solver, '../output/mdvrp_detailed_routes.csv')
        
        # Visualize routes
        visualize_choice = input("Visualize routes? (y/n, default y): ") or "y"
        if visualize_choice.lower() == 'y':
            solver.visualize_routes(solution)
        
        # Print route details
        print_route_details(solution, solver)

def create_detailed_output(solution, solver, filename):
    """Create detailed CSV output with route information"""
    detailed_data = []
    
    for route in solution['routes']:
        for i, stop_idx in enumerate(route['stops']):
            stop_info = solver.get_stop_info(stop_idx)
            detailed_data.append({
                'vehicle_id': route['vehicle_id'],
                'stop_sequence': i,
                'stop_index': stop_idx,
                'cluster_number': stop_info.get('cluster_number', stop_idx),
                'latitude': stop_info.get('latitude', 0),
                'longitude': stop_info.get('longitude', 0),
                'students': stop_info.get('students', 0),
                'route_name': stop_info.get('route_name', ''),
                'route_total_distance': route['distance'],
                'route_total_students': route['students'],
                'is_depot': stop_idx in solver.depot_indices
            })
    
    df = pd.DataFrame(detailed_data)
    df.to_csv(filename, index=False)
    print(f"Detailed routes saved to {filename}")

def print_route_details(solution, solver):
    """Print detailed route information"""
    print(f"\n=== DETAILED ROUTE INFORMATION ===")
    
    for route in solution['routes']:
        print(f"\n--- Vehicle {route['vehicle_id']} ---")
        print(f"Total Distance: {route['distance']:,} meters ({route['distance']/1000:.1f} km)")
        print(f"Total Students: {route['students']}")
        print(f"Capacity Utilization: {route['students']/solver.vehicle_capacity*100:.1f}%")
        print("Route Details:")
        
        for i, stop_idx in enumerate(route['stops']):
            stop_info = solver.get_stop_info(stop_idx)
            stop_type = "DEPOT" if stop_idx in solver.depot_indices else "STOP"
            print(f"  {i+1:2d}. [{stop_type}] Cluster {stop_info['cluster_number']:3d} - "
                  f"{stop_info['route_name'][:50]:50s} - {stop_info['students']:2d} students")

if __name__ == "__main__":
    main()