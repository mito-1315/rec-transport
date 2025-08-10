import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import seaborn as sns

class DistanceMatrixMDVRP:
    def __init__(self, distance_matrix_file, stops_data_file, depot_allocation=None):
        """Initialize MDVRP solver using pre-computed distance matrix"""
        self.load_distance_matrix(distance_matrix_file)
        self.load_stops_data(stops_data_file)
        self.setup_problem_parameters(depot_allocation)
        
    def load_distance_matrix(self, distance_matrix_file):
        """Load pre-computed distance matrix"""
        print("="*60)
        print("LOADING PRE-COMPUTED DISTANCE MATRIX")
        print("="*60)
        
        # Load distance matrix
        self.distance_df = pd.read_csv(distance_matrix_file, index_col=0)
        self.distance_matrix = self.distance_df.values.astype(int)
        
        print(f"Distance matrix loaded:")
        print(f"- Shape: {self.distance_matrix.shape}")
        print(f"- Total locations: {len(self.distance_matrix)}")
        print(f"- Matrix file: {distance_matrix_file}")
        
        # Verify matrix is symmetric and has zeros on diagonal
        if not np.allclose(self.distance_matrix, self.distance_matrix.T, rtol=1e-5):
            print("⚠ Warning: Distance matrix is not symmetric")
        
        if not np.all(np.diag(self.distance_matrix) == 0):
            print("⚠ Warning: Diagonal is not zero")
            
    def load_stops_data(self, stops_data_file):
        """Load stops data with coordinates and student counts"""
        print(f"\nLoading stops data from: {stops_data_file}")
        
        self.stops_df = pd.read_csv(stops_data_file)
        
        print(f"Stops data loaded:")
        print(f"- Total stops: {len(self.stops_df)}")
        
        # Verify data consistency
        if len(self.stops_df) != len(self.distance_matrix):
            print(f"❌ ERROR: Mismatch between distance matrix ({len(self.distance_matrix)}) and stops data ({len(self.stops_df)})")
            raise ValueError("Distance matrix and stops data have different sizes")
        
        # Analyze the data
        depots = self.stops_df[self.stops_df['is_depot'] == True]
        school = self.stops_df[self.stops_df['is_school'] == True]
        stops = self.stops_df[(self.stops_df['is_depot'] == False) & (self.stops_df['is_school'] == False)]
        
        print(f"- Depots: {len(depots)}")
        print(f"- Regular stops: {len(stops)}")
        print(f"- School destinations: {len(school)}")
        print(f"- Total students: {stops['num_students'].sum()}")
        
        # Get indices
        self.depot_indices = depots.index.tolist()
        self.school_indices = school.index.tolist()
        self.stop_indices = stops.index.tolist()
        
        print(f"\nLocation indices:")
        print(f"- Depot indices: {self.depot_indices}")
        print(f"- School indices: {self.school_indices}")
        print(f"- First few stop indices: {self.stop_indices[:10]}...")
        
    def setup_problem_parameters(self, depot_allocation=None):
        """Setup MDVRP parameters"""
        print(f"\n{'='*60}")
        print("SETTING UP PROBLEM PARAMETERS")
        print("="*60)
        
        # Auto-detect or use provided depot allocation
        if depot_allocation is None:
            # Auto-generate depot allocation based on number of depots
            num_depots = len(self.depot_indices)
            total_students = self.stops_df['num_students'].sum()
            buses_needed = int(np.ceil(total_students / 55))  # 55 capacity per bus
            
            # Distribute buses among depots
            base_buses = buses_needed // num_depots
            extra_buses = buses_needed % num_depots
            
            self.depot_bus_allocation = {}
            for i, depot_idx in enumerate(self.depot_indices):
                buses = base_buses + (1 if i < extra_buses else 0)
                self.depot_bus_allocation[depot_idx] = max(1, buses)  # At least 1 bus per depot
                
            print(f"Auto-generated depot allocation:")
        else:
            self.depot_bus_allocation = depot_allocation
            print(f"Using provided depot allocation:")
            
        for depot_idx, buses in self.depot_bus_allocation.items():
            depot_name = self.stops_df.iloc[depot_idx]['stop_name']
            print(f"  Depot {depot_idx} ({depot_name}): {buses} buses")
        
        self.vehicle_capacity = 55
        self.total_vehicles = sum(self.depot_bus_allocation.values())
        
        # Create vehicle start and end points
        self.vehicle_starts = []
        self.vehicle_ends = []
        
        # Assume single school destination (modify if multiple schools)
        school_index = self.school_indices[0] if self.school_indices else len(self.stops_df) - 1
        
        for depot_idx, num_buses in self.depot_bus_allocation.items():
            for _ in range(num_buses):
                self.vehicle_starts.append(depot_idx)
                self.vehicle_ends.append(school_index)
        
        print(f"\nVehicle Configuration:")
        print(f"- Total vehicles: {self.total_vehicles}")
        print(f"- Vehicle capacity: {self.vehicle_capacity} students")
        print(f"- School destination index: {school_index}")
        
        total_students = self.stops_df['num_students'].sum()
        total_capacity = self.total_vehicles * self.vehicle_capacity
        print(f"- Total students: {total_students}")
        print(f"- Total capacity: {total_capacity}")
        print(f"- Capacity utilization: {total_students/total_capacity*100:.1f}%")
        
    def solve_mdvrp(self, time_limit_seconds=600):
        """Solve the MDVRP problem using the distance matrix"""
        print(f"\n{'='*60}")
        print("SOLVING MDVRP WITH DISTANCE MATRIX")
        print(f"{'='*60}")
        
        # Prepare demands
        demands = self.stops_df['num_students'].tolist()
        
        # Create data model
        data = {
            'distance_matrix': self.distance_matrix,
            'demands': demands,
            'vehicle_capacities': [self.vehicle_capacity] * self.total_vehicles,
            'num_vehicles': self.total_vehicles,
            'starts': self.vehicle_starts,
            'ends': self.vehicle_ends
        }
        
        print(f"Problem setup:")
        print(f"- Matrix size: {len(self.distance_matrix)}x{len(self.distance_matrix)}")
        print(f"- Vehicles: {data['num_vehicles']}")
        print(f"- Total demand: {sum(demands)} students")
        print(f"- Time limit: {time_limit_seconds} seconds")
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']),
            data['num_vehicles'],
            data['starts'],
            data['ends']
        )
        
        routing = pywrapcp.RoutingModel(manager)
        
        # Distance callback using pre-computed matrix
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
        
        print(f"Starting optimization...")
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self.process_solution(data, manager, routing, solution)
        else:
            print("❌ No solution found!")
            return None
            
    def process_solution(self, data, manager, routing, solution):
        """Process and display solution"""
        print(f"\n{'='*50}")
        print("✅ SOLUTION FOUND!")
        print(f"{'='*50}")
        print(f"Objective (Total Distance): {solution.ObjectiveValue():,} meters")
        print(f"Objective (Total Distance): {solution.ObjectiveValue()/1000:.1f} km")
        
        routes = []
        total_distance = 0
        total_students = 0
        
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
            
            # Add final destination
            final_node = manager.IndexToNode(index)
            route_stops.append(final_node)
            
            if len(route_stops) > 2:  # Routes with actual stops
                route_info = {
                    'vehicle_id': vehicle_id,
                    'stops': route_stops,
                    'distance_m': route_distance,
                    'distance_km': route_distance / 1000,
                    'students': route_students,
                    'capacity_utilization': (route_students / self.vehicle_capacity) * 100,
                    'start_depot': route_stops[0],
                    'end_school': route_stops[-1],
                    'num_pickup_stops': len(route_stops) - 2
                }
                routes.append(route_info)
                
                total_distance += route_distance
                total_students += route_students
                
                # Print route summary
                depot_name = self.stops_df.iloc[route_stops[0]]['stop_name']
                school_name = self.stops_df.iloc[route_stops[-1]]['stop_name']
                
                print(f"\n--- Bus {vehicle_id} ---")
                print(f"Route: {depot_name} → ... → {school_name}")
                print(f"Stops: {len(route_stops)} total ({route_info['num_pickup_stops']} pickups)")
                print(f"Distance: {route_distance:,}m ({route_distance/1000:.1f}km)")
                print(f"Students: {route_students}/{self.vehicle_capacity} ({route_students/self.vehicle_capacity*100:.1f}%)")
        
        # Summary
        print(f"\n{'='*50}")
        print("SOLUTION SUMMARY")
        print(f"{'='*50}")
        print(f"Total Distance: {total_distance:,} meters ({total_distance/1000:.1f} km)")
        print(f"Total Students: {total_students}")
        print(f"Active Vehicles: {len(routes)}/{self.total_vehicles}")
        print(f"Average Distance per Vehicle: {total_distance/len(routes):,.0f} meters")
        print(f"Average Students per Vehicle: {total_students/len(routes):.1f}")
        
        # Save solution
        self.save_solution(routes)
        
        return {
            'routes': routes,
            'total_distance': total_distance,
            'total_students': total_students,
            'vehicles_used': len(routes),
            'objective_value': solution.ObjectiveValue()
        }
        
    def save_solution(self, routes):
        """Save detailed solution to CSV"""
        detailed_data = []
        
        for route in routes:
            for seq, stop_idx in enumerate(route['stops']):
                stop_info = self.stops_df.iloc[stop_idx]
                detailed_data.append({
                    'bus_id': route['vehicle_id'],
                    'sequence': seq,
                    'stop_index': stop_idx,
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
                    'route_capacity_utilization': route['capacity_utilization']
                })
        
        solution_df = pd.DataFrame(detailed_data)
        solution_df.to_csv('mdvrp_solution_with_distance_matrix.csv', index=False)
        print(f"\n💾 Solution saved to 'mdvrp_solution_with_distance_matrix.csv'")

def main():
    """Main function to run MDVRP with distance matrix"""
    
    # Configuration
    distance_matrix_file = './inputs/distMatrixWithDepot.csv'
    stops_data_file = './inputs/mdvrp_data.csv'  # Your stops data with coordinates and student counts

    # Optional: Custom depot allocation (if you want to override auto-detection)
    # custom_depot_allocation = {
    #     0: 5,  # Depot at index 0: 5 buses
    #     1: 3,  # Depot at index 1: 3 buses
    #     # ... add more as needed
    # }
    
    try:
        print("🚌 Starting MDVRP with Distance Matrix...")
        
        # Create solver
        solver = DistanceMatrixMDVRP(
            distance_matrix_file=distance_matrix_file,
            stops_data_file=stops_data_file,
            depot_allocation=None  # Use None for auto-detection or pass custom_depot_allocation
        )
        
        # Solve problem
        solution = solver.solve_mdvrp(time_limit_seconds=600)
        
        if solution:
            print(f"\n🎉 SUCCESS!")
            print(f"✅ Routed {solution['total_students']} students")
            print(f"✅ Used {solution['vehicles_used']} buses")
            print(f"✅ Total distance: {solution['total_distance']/1000:.1f} km")
        else:
            print(f"\n❌ FAILED to find solution")
            
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please ensure the following files exist:")
        print(f"1. {distance_matrix_file}")
        print(f"2. {stops_data_file}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()