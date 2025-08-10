import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import json
import time
from typing import List, Dict, Tuple
import logging

# =============================================================================
# CONFIGURATION PARAMETERS - Adjust these values as needed
# =============================================================================

# File paths
DISTANCE_MATRIX_FILE = "distance-matrix.csv"
DEPOT_STOP_DATA_FILE = "depot-stop-data.csv"
OUTPUT_ROUTES_FILE = "optimized_routes.json"
OUTPUT_SUMMARY_FILE = "route_summary.csv"

# Vehicle and capacity constraints
MAX_VEHICLE_CAPACITY = 55  # Maximum students per bus
TOTAL_BUSES_AVAILABLE = 135  # Total number of buses (130-140 range)
MAX_BUSES_PER_DEPOT = 10  # Maximum buses that can start from a single depot

# Optimization settings
SEARCH_TIME_LIMIT_SECONDS = 120  # Maximum time for optimization (2 minutes)
FIRST_SOLUTION_STRATEGY = routing_enums_pb2.FirstSolutionStrategy.SAVINGS
LOCAL_SEARCH_METAHEURISTIC = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

# Penalties and costs
PENALTY_FOR_UNSERVED_STOP = 10000  # High penalty for not serving a stop
DEPOT_PENALTY = 100  # Small penalty to balance depot usage

# Distance scaling factor (if distances are in different units)
DISTANCE_SCALE_FACTOR = 0.1

# Logging level
LOG_LEVEL = logging.INFO

# =============================================================================

class CollegeBusRouteOptimizer:
    def __init__(self):
        logging.basicConfig(level=LOG_LEVEL)
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.distance_matrix = None
        self.depot_stop_data = None
        self.depots = []
        self.stops = []
        self.college_id = None
        self.demands = []
        self.virtual_stops = []  # For handling stops with demand > capacity
        
        # OR-Tools objects
        self.manager = None
        self.routing = None
        self.solution = None
        
    def load_data(self):
        """Load distance matrix and depot-stop data from CSV files."""
        try:
            self.logger.info("Loading data files...")
            
            # Load distance matrix
            self.distance_matrix = pd.read_csv(DISTANCE_MATRIX_FILE, index_col=0)
            self.logger.info(f"Loaded distance matrix: {self.distance_matrix.shape}")
            
            # Load depot and stop data
            self.depot_stop_data = pd.read_csv(DEPOT_STOP_DATA_FILE)
            self.logger.info(f"Loaded {len(self.depot_stop_data)} locations")
            
            # Identify depots, stops, and college
            self.depots = self.depot_stop_data[self.depot_stop_data['is_depot'] == True]['stop_id'].tolist()
            self.stops = self.depot_stop_data[self.depot_stop_data['is_depot'] == False]['stop_id'].tolist()
            
            # Find college (assuming it's marked as is_school = True)
            college_rows = self.depot_stop_data[self.depot_stop_data['is_school'] == True]
            if len(college_rows) > 0:
                self.college_id = college_rows.iloc[0]['stop_id']
            else:
                # If no college marked, assume it's the last stop
                self.college_id = self.stops[-1]
                self.logger.warning(f"No college marked, assuming {self.college_id} is the college")
            
            self.logger.info(f"Found {len(self.depots)} depots, {len(self.stops)} stops, college: {self.college_id}")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self):
        """Handle stops with demand > capacity by creating virtual stops."""
        self.logger.info("Preprocessing data...")
        
        self.virtual_stops = []
        processed_demands = []
        
        for _, row in self.depot_stop_data.iterrows():
            stop_id = row['stop_id']
            demand = row['num_students'] if not pd.isna(row['num_students']) else 0
            
            if row['is_depot'] or stop_id == self.college_id:
                # Depots and college have 0 demand
                processed_demands.append(0)
            elif demand > MAX_VEHICLE_CAPACITY:
                # Split high-demand stops into virtual stops
                num_virtual_stops = int(np.ceil(demand / MAX_VEHICLE_CAPACITY))
                remaining_demand = demand
                
                for i in range(num_virtual_stops):
                    virtual_demand = min(MAX_VEHICLE_CAPACITY, remaining_demand)
                    processed_demands.append(virtual_demand)
                    
                    if i > 0:  # First virtual stop uses original stop_id
                        virtual_stop_id = f"{stop_id}_v{i}"
                        self.virtual_stops.append({
                            'original_id': stop_id,
                            'virtual_id': virtual_stop_id,
                            'demand': virtual_demand
                        })
                    
                    remaining_demand -= virtual_demand
                    
                self.logger.info(f"Split stop {stop_id} (demand: {demand}) into {num_virtual_stops} virtual stops")
            else:
                processed_demands.append(int(demand))
        
        self.demands = processed_demands
        self.logger.info(f"Processed demands: {len(self.demands)} locations")
    
    def create_distance_callback(self):
        """Create distance callback function for OR-Tools."""
        distance_matrix_values = self.distance_matrix.values * DISTANCE_SCALE_FACTOR
        
        def distance_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return int(distance_matrix_values[from_node][to_node])
        
        return distance_callback
    
    def create_demand_callback(self):
        """Create demand callback function for OR-Tools."""
        def demand_callback(from_index):
            from_node = self.manager.IndexToNode(from_index)
            return self.demands[from_node]
        
        return demand_callback
    
    def setup_model(self):
        """Set up the OR-Tools VRP model."""
        self.logger.info("Setting up VRP model...")
        
        # Get college index
        college_index = self.distance_matrix.index.get_loc(self.college_id)
        
        # Create routing index manager
        num_locations = len(self.distance_matrix)
        num_depots = len(self.depots)
        
        # Depot indices
        depot_indices = [self.distance_matrix.index.get_loc(depot) for depot in self.depots]
        
        # Calculate number of vehicles needed (estimate)
        total_demand = sum(self.demands)
        estimated_vehicles = min(int(np.ceil(total_demand / MAX_VEHICLE_CAPACITY)), TOTAL_BUSES_AVAILABLE)
        
        self.logger.info(f"Total demand: {total_demand}, Estimated vehicles needed: {estimated_vehicles}")
        
        # Distribute vehicles across depots
        vehicles_per_depot = max(1, estimated_vehicles // num_depots)
        extra_vehicles = estimated_vehicles % num_depots
        
        start_depots = []
        end_destinations = []
        
        for i, depot_idx in enumerate(depot_indices):
            # Number of vehicles for this depot
            depot_vehicles = vehicles_per_depot + (1 if i < extra_vehicles else 0)
            
            # Add vehicles starting from this depot
            for _ in range(depot_vehicles):
                start_depots.append(depot_idx)
                end_destinations.append(college_index)
        
        # Adjust total vehicles to match what we actually created
        actual_vehicles = len(start_depots)
        self.logger.info(f"Using {actual_vehicles} vehicles distributed across {num_depots} depots")
        self.logger.info(f"Vehicles per depot: {vehicles_per_depot} (+ {extra_vehicles} depots get 1 extra)")
        
        # Create manager with distributed vehicles
        self.manager = pywrapcp.RoutingIndexManager(
            num_locations,
            actual_vehicles,
            start_depots,  # Start depots (distributed)
            end_destinations  # All end at college
        )
        
        # Create routing model
        self.routing = pywrapcp.RoutingModel(self.manager)
        
        # Create and register distance callback
        distance_callback = self.create_distance_callback()
        transit_callback_index = self.routing.RegisterTransitCallback(distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Create and register demand callback
        demand_callback = self.create_demand_callback()
        demand_callback_index = self.routing.RegisterUnaryTransitCallback(demand_callback)
        
        # Add capacity constraint
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [MAX_VEHICLE_CAPACITY] * actual_vehicles,  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity'
        )
        
        # Add penalty for unserved locations (except depots and college)
        for node in range(num_locations):
            if (node not in depot_indices and 
                node != college_index and 
                self.demands[node] > 0):
                self.routing.AddDisjunction([self.manager.NodeToIndex(node)], PENALTY_FOR_UNSERVED_STOP)
    
    def solve(self):
        """Solve the VRP problem."""
        self.logger.info("Starting optimization...")
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = FIRST_SOLUTION_STRATEGY
        search_parameters.local_search_metaheuristic = LOCAL_SEARCH_METAHEURISTIC
        search_parameters.time_limit.seconds = SEARCH_TIME_LIMIT_SECONDS
        search_parameters.log_search = True
        
        # Solve
        start_time = time.time()
        self.solution = self.routing.SolveWithParameters(search_parameters)
        solve_time = time.time() - start_time
        
        if self.solution:
            self.logger.info(f"Solution found in {solve_time:.2f} seconds!")
            self.logger.info(f"Objective value: {self.solution.ObjectiveValue()}")
        else:
            self.logger.error("No solution found!")
            return False
        
        return True
    
    def extract_routes(self):
        """Extract routes from the solution."""
        if not self.solution:
            return None
        
        routes = []
        total_distance = 0
        total_load = 0
        depot_usage = {}
        
        for vehicle_id in range(self.routing.vehicles()):
            index = self.routing.Start(vehicle_id)
            route = {
                'vehicle_id': vehicle_id,
                'stops': [],
                'distance': 0,
                'load': 0,
                'depot': None
            }
            
            route_distance = 0
            route_load = 0
            
            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                stop_id = self.distance_matrix.index[node_index]
                demand = self.demands[node_index]
                
                # First stop is the depot
                if len(route['stops']) == 0:
                    route['depot'] = stop_id
                    depot_usage[stop_id] = depot_usage.get(stop_id, 0) + 1
                
                route['stops'].append({
                    'stop_id': stop_id,
                    'demand': demand,
                    'cumulative_load': route_load + demand
                })
                
                route_load += demand
                previous_index = index
                index = self.solution.Value(self.routing.NextVar(index))
                
                if not self.routing.IsEnd(index):
                    route_distance += self.routing.GetArcCostForVehicle(
                        previous_index, index, vehicle_id)
            
            # Add final distance to college
            final_node_index = self.manager.IndexToNode(index)
            route_distance += self.routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
            
            route['distance'] = route_distance / DISTANCE_SCALE_FACTOR  # Convert back to original units
            route['load'] = route_load
            
            # Only add routes that have stops (beyond depot and college)
            if len(route['stops']) > 1:  # More than just the depot
                routes.append(route)
                total_distance += route['distance']
                total_load += route_load
        
        return {
            'routes': routes,
            'depot_usage': depot_usage,
            'summary': {
                'total_routes': len(routes),
                'total_distance': total_distance,
                'total_students': total_load,
                'avg_distance_per_route': total_distance / len(routes) if routes else 0,
                'avg_students_per_bus': total_load / len(routes) if routes else 0,
                'depots_used': len(depot_usage),
                'max_vehicles_from_depot': max(depot_usage.values()) if depot_usage else 0
            }
        }
    
    def save_results(self, results):
        """Save optimization results to files."""
        if not results:
            self.logger.error("No results to save!")
            return
        
        try:
            # Save detailed routes
            with open(OUTPUT_ROUTES_FILE, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Routes saved to {OUTPUT_ROUTES_FILE}")
            
            # Save summary
            summary_data = []
            for i, route in enumerate(results['routes']):
                summary_data.append({
                    'route_id': i + 1,
                    'vehicle_id': route['vehicle_id'],
                    'depot': route['depot'],
                    'num_stops': len(route['stops']) - 1,  # Exclude depot
                    'total_students': route['load'],
                    'total_distance': route['distance'],
                    'stops_sequence': ' -> '.join([stop['stop_id'] for stop in route['stops']]) + f" -> {self.college_id}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(OUTPUT_SUMMARY_FILE, index=False)
            self.logger.info(f"Summary saved to {OUTPUT_SUMMARY_FILE}")
            
            # Print summary
            self.print_summary(results)
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def print_summary(self, results):
        """Print optimization summary."""
        print("\n" + "="*60)
        print("COLLEGE BUS ROUTE OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Total Routes: {results['summary']['total_routes']}")
        print(f"Total Students: {results['summary']['total_students']}")
        print(f"Total Distance: {results['summary']['total_distance']:.2f} meters")
        print(f"Average Distance per Route: {results['summary']['avg_distance_per_route']:.2f} meters")
        print(f"Average Students per Bus: {results['summary']['avg_students_per_bus']:.2f}")
        print(f"Depots Used: {results['summary']['depots_used']}/24")
        print(f"Max Vehicles from Single Depot: {results['summary']['max_vehicles_from_depot']}")
        print("\nDepot Usage:")
        for depot, count in results['depot_usage'].items():
            print(f"  {depot}: {count} vehicles")
        print("="*60)
        
        # Show first few routes
        for i, route in enumerate(results['routes'][:5]):
            print(f"\nRoute {i+1} (Vehicle {route['vehicle_id']}):")
            print(f"  Depot: {route['depot']}")
            print(f"  Students: {route['load']}")
            print(f"  Distance: {route['distance']:.2f} meters")
            stops_str = " -> ".join([stop['stop_id'] for stop in route['stops']])
            print(f"  Path: {stops_str} -> {self.college_id}")
        
        if len(results['routes']) > 5:
            print(f"\n... and {len(results['routes']) - 5} more routes")
    
    def run_optimization(self):
        """Run the complete optimization process."""
        try:
            self.load_data()
            self.preprocess_data()
            self.setup_model()
            
            if self.solve():
                results = self.extract_routes()
                self.save_results(results)
                return results
            else:
                self.logger.error("Optimization failed!")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during optimization: {e}")
            raise

def main():
    """Main function to run the bus route optimization."""
    optimizer = CollegeBusRouteOptimizer()
    results = optimizer.run_optimization()
    
    if results:
        print("\nOptimization completed successfully!")
        print(f"Check '{OUTPUT_ROUTES_FILE}' for detailed routes")
        print(f"Check '{OUTPUT_SUMMARY_FILE}' for route summary")
    else:
        print("Optimization failed!")

if __name__ == "__main__":
    main()