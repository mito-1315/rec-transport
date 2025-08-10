import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import seaborn as sns

class OptimizedCapacityMDVRP:
    def __init__(self, distance_matrix_file, stops_data_file, depot_allocation=None):
        """Initialize MDVRP solver with optimized capacity utilization"""
        self.load_distance_matrix(distance_matrix_file)
        self.load_stops_data(stops_data_file)
        self.setup_problem_parameters(depot_allocation)
        
    def load_distance_matrix(self, distance_matrix_file):
        """Load pre-computed distance matrix"""
        print("="*60)
        print("LOADING PRE-COMPUTED DISTANCE MATRIX")
        print("="*60)
        
        self.distance_df = pd.read_csv(distance_matrix_file, index_col=0)
        self.distance_matrix = self.distance_df.values.astype(int)
        
        print(f"Distance matrix loaded:")
        print(f"- Shape: {self.distance_matrix.shape}")
        print(f"- Total locations: {len(self.distance_matrix)}")
        
    def load_stops_data(self, stops_data_file):
        """Load stops data with coordinates and student counts"""
        print(f"\nLoading stops data from: {stops_data_file}")
        
        self.stops_df = pd.read_csv(stops_data_file)
        
        # Verify data consistency
        if len(self.stops_df) != len(self.distance_matrix):
            raise ValueError("Distance matrix and stops data have different sizes")
        
        # Analyze the data
        depots = self.stops_df[self.stops_df['is_depot'] == True]
        school = self.stops_df[self.stops_df['is_school'] == True]
        stops = self.stops_df[(self.stops_df['is_depot'] == False) & (self.stops_df['is_school'] == False)]
        
        print(f"- Total stops: {len(self.stops_df)}")
        print(f"- Depots: {len(depots)}")
        print(f"- Regular stops: {len(stops)}")
        print(f"- School destinations: {len(school)}")
        print(f"- Total students: {stops['num_students'].sum()}")
        
        # Get indices
        self.depot_indices = depots.index.tolist()
        self.school_indices = school.index.tolist()
        self.stop_indices = stops.index.tolist()
        
    def setup_problem_parameters(self, depot_allocation=None):
        """Setup MDVRP parameters targeting 90% capacity utilization"""
        print(f"\n{'='*60}")
        print("SETTING UP OPTIMIZED CAPACITY PARAMETERS")
        print("="*60)
        
        total_students = self.stops_df['num_students'].sum()
        
        # **KEY CHANGE: Target 90% capacity utilization**
        self.vehicle_capacity = 55  # Keep standard bus capacity
        target_utilization = 0.90  # 90% utilization target
        
        # Calculate buses needed for 90% utilization
        effective_capacity_per_bus = int(self.vehicle_capacity * target_utilization)  # 49.5 ≈ 49
        buses_needed = int(np.ceil(total_students / effective_capacity_per_bus))
        
        print(f"Capacity Planning:")
        print(f"- Bus capacity: {self.vehicle_capacity} students")
        print(f"- Target utilization: {target_utilization * 100}%")
        print(f"- Effective capacity per bus: {effective_capacity_per_bus} students")
        print(f"- Total students: {total_students}")
        print(f"- Buses needed for 90% utilization: {buses_needed}")
        
        if depot_allocation is None:
            # **ENHANCED DEPOT ALLOCATION: Use more depots with more buses**
            
            # Use more major depots to distribute load better
            num_major_depots = min(15, len(self.depot_indices))  # Increased from 10 to 15
            major_depot_indices = self.depot_indices[:num_major_depots]
            
            # Calculate buses per depot ensuring we have enough total capacity
            min_buses_per_depot = 3  # Minimum buses per depot
            base_buses = max(min_buses_per_depot, buses_needed // num_major_depots)
            extra_buses = buses_needed % num_major_depots
            
            # Add extra buses to ensure sufficient capacity
            safety_margin = int(buses_needed * 0.15)  # 15% safety margin
            total_planned_buses = base_buses * num_major_depots + extra_buses + safety_margin
            
            self.depot_bus_allocation = {}
            extra_buses_distributed = 0
            
            for i, depot_idx in enumerate(major_depot_indices):
                buses = base_buses
                
                # Distribute extra buses
                if extra_buses_distributed < extra_buses:
                    buses += 1
                    extra_buses_distributed += 1
                
                # Add safety margin buses to larger depots
                if i < safety_margin:
                    buses += 1
                
                self.depot_bus_allocation[depot_idx] = buses
                
            print(f"\nUsing {num_major_depots} major depots with enhanced allocation:")
        else:
            self.depot_bus_allocation = depot_allocation
            print(f"Using provided depot allocation:")
            
        total_allocated_buses = 0
        for depot_idx, buses in self.depot_bus_allocation.items():
            depot_name = self.stops_df.iloc[depot_idx]['stop_name']
            total_allocated_buses += buses
            print(f"  Depot {depot_idx} ({depot_name}): {buses} buses")
        
        self.total_vehicles = total_allocated_buses
        
        # Calculate actual capacity and utilization
        total_capacity = self.total_vehicles * self.vehicle_capacity
        actual_utilization = total_students / total_capacity
        
        print(f"\nCapacity Analysis:")
        print(f"- Total vehicles allocated: {self.total_vehicles}")
        print(f"- Total theoretical capacity: {total_capacity} students")
        print(f"- Total students to transport: {total_students}")
        print(f"- Actual capacity utilization: {actual_utilization * 100:.1f}%")
        print(f"- Safety margin: {(1 - actual_utilization) * 100:.1f}%")
        
        # Ensure we're not over-utilizing
        if actual_utilization > 0.95:
            print("⚠️  WARNING: Utilization > 95%. Adding more buses...")
            additional_buses_needed = int(np.ceil((total_students / 0.90 - total_capacity) / self.vehicle_capacity))
            
            # Add buses to largest depots
            depot_indices_sorted = sorted(self.depot_bus_allocation.keys(), 
                                        key=lambda x: self.depot_bus_allocation[x], reverse=True)
            
            for i in range(additional_buses_needed):
                depot_to_add = depot_indices_sorted[i % len(depot_indices_sorted)]
                self.depot_bus_allocation[depot_to_add] += 1
                self.total_vehicles += 1
            
            # Recalculate
            total_capacity = self.total_vehicles * self.vehicle_capacity
            actual_utilization = total_students / total_capacity
            print(f"✅ Adjusted: {self.total_vehicles} vehicles, {actual_utilization * 100:.1f}% utilization")
        
        # Create vehicle start and end points
        self.vehicle_starts = []
        self.vehicle_ends = []
        
        school_index = self.school_indices[0] if self.school_indices else len(self.stops_df) - 1
        
        for depot_idx, num_buses in self.depot_bus_allocation.items():
            for _ in range(num_buses):
                self.vehicle_starts.append(depot_idx)
                self.vehicle_ends.append(school_index)
        
        print(f"\nFinal Vehicle Configuration:")
        print(f"- Total vehicles: {self.total_vehicles}")
        print(f"- Vehicle capacity: {self.vehicle_capacity} students")
        print(f"- School destination index: {school_index}")
        print(f"- Target utilization per bus: ≤ 90%")
        print(f"- Expected students per bus: ~{total_students / self.total_vehicles:.1f}")
        
    def solve_mdvrp(self, time_limit_seconds=1800):
        """Solve the MDVRP problem with optimized capacity"""
        print(f"\n{'='*60}")
        print("SOLVING MDVRP WITH OPTIMIZED CAPACITY")
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
        print(f"- Time limit: {time_limit_seconds} seconds ({time_limit_seconds/60:.1f} minutes)")
        
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
        
        # **CAPACITY CONSTRAINT: Generous slack for 90% target**
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        # Allow up to 90% of capacity (with some flexibility)
        max_capacity_per_vehicle = int(self.vehicle_capacity * 0.92)  # 92% to allow slight flexibility
        
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # No slack - we have enough vehicles
            [max_capacity_per_vehicle] * self.total_vehicles,  # Reduced capacity limit
            True,
            'Capacity'
        )
        
        print(f"Capacity constraint: Max {max_capacity_per_vehicle} students per bus (≈90% of {self.vehicle_capacity})")
        
        # **OPTIMIZED SEARCH PARAMETERS**
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(time_limit_seconds)
        search_parameters.log_search = True
        
        print(f"Starting optimization...")
        print(f"Using PARALLEL_CHEAPEST_INSERTION for initial solution...")
        
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            print(f"\n✅ SOLUTION FOUND!")
            return self.process_solution(data, manager, routing, solution)
        else:
            print(f"❌ No solution found after {time_limit_seconds} seconds")
            print("\nTroubleshooting suggestions:")
            print("1. Increase time limit further")
            print("2. Add more vehicles")
            print("3. Check if distance matrix has unreachable locations")
            return None
            
    def process_solution(self, data, manager, routing, solution):
        """Process and display solution with capacity analysis"""
        print(f"\n{'='*50}")
        print("✅ SOLUTION FOUND!")
        print(f"{'='*50}")
        print(f"Objective (Total Distance): {solution.ObjectiveValue():,} meters")
        print(f"Objective (Total Distance): {solution.ObjectiveValue()/1000:.1f} km")
        
        routes = []
        total_distance = 0
        total_students = 0
        capacity_violations = 0
        max_utilization = 0
        
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
                utilization = (route_students / self.vehicle_capacity) * 100
                max_utilization = max(max_utilization, utilization)
                
                if utilization > 90:
                    capacity_violations += 1
                
                route_info = {
                    'vehicle_id': vehicle_id,
                    'stops': route_stops,
                    'distance_m': route_distance,
                    'distance_km': route_distance / 1000,
                    'students': route_students,
                    'capacity_utilization': utilization,
                    'start_depot': route_stops[0],
                    'end_school': route_stops[-1],
                    'num_pickup_stops': len(route_stops) - 2
                }
                routes.append(route_info)
                
                total_distance += route_distance
                total_students += route_students
                
                # Print route summary with capacity focus
                depot_name = self.stops_df.iloc[route_stops[0]]['stop_name']
                school_name = self.stops_df.iloc[route_stops[-1]]['stop_name']
                
                utilization_status = "✅" if utilization <= 90 else "⚠️"
                
                print(f"\n--- Bus {vehicle_id} {utilization_status} ---")
                print(f"Route: {depot_name} → ... → {school_name}")
                print(f"Stops: {len(route_stops)} total ({route_info['num_pickup_stops']} pickups)")
                print(f"Distance: {route_distance:,}m ({route_distance/1000:.1f}km)")
                print(f"Students: {route_students}/{self.vehicle_capacity} ({utilization:.1f}%)")
        
        # Detailed Summary
        print(f"\n{'='*50}")
        print("SOLUTION SUMMARY")
        print(f"{'='*50}")
        print(f"Total Distance: {total_distance:,} meters ({total_distance/1000:.1f} km)")
        print(f"Total Students Transported: {total_students}")
        print(f"Active Vehicles: {len(routes)}/{self.total_vehicles}")
        print(f"Unused Vehicles: {self.total_vehicles - len(routes)}")
        
        if len(routes) > 0:
            avg_utilization = (total_students / (len(routes) * self.vehicle_capacity)) * 100
            print(f"\n📊 CAPACITY ANALYSIS:")
            print(f"- Average utilization per active bus: {avg_utilization:.1f}%")
            print(f"- Maximum utilization: {max_utilization:.1f}%")
            print(f"- Buses over 90% capacity: {capacity_violations}/{len(routes)}")
            print(f"- Average students per bus: {total_students/len(routes):.1f}")
            print(f"- Average distance per bus: {total_distance/len(routes):,.0f} meters")
            
            # Utilization distribution
            utilizations = [r['capacity_utilization'] for r in routes]
            print(f"- Utilization range: {min(utilizations):.1f}% - {max(utilizations):.1f}%")
            
            under_50 = sum(1 for u in utilizations if u < 50)
            between_50_75 = sum(1 for u in utilizations if 50 <= u < 75)
            between_75_90 = sum(1 for u in utilizations if 75 <= u <= 90)
            over_90 = sum(1 for u in utilizations if u > 90)
            
            print(f"- Distribution:")
            print(f"  • Under 50%: {under_50} buses")
            print(f"  • 50-75%: {between_50_75} buses")
            print(f"  • 75-90%: {between_75_90} buses ✅ TARGET")
            print(f"  • Over 90%: {over_90} buses ⚠️")
        
        # Save solution
        self.save_solution(routes)
        
        return {
            'routes': routes,
            'total_distance': total_distance,
            'total_students': total_students,
            'vehicles_used': len(routes),
            'average_utilization': avg_utilization if len(routes) > 0 else 0,
            'max_utilization': max_utilization,
            'capacity_violations': capacity_violations,
            'objective_value': solution.ObjectiveValue()
        }
        
    def save_solution(self, routes):
        """Save detailed solution to CSV with capacity analysis"""
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
                    'route_capacity_utilization': route['capacity_utilization'],
                    'target_utilization_met': route['capacity_utilization'] <= 90
                })
        
        solution_df = pd.DataFrame(detailed_data)
        solution_df.to_csv('mdvrp_optimized_capacity_solution.csv', index=False)
        print(f"\n💾 Solution saved to 'mdvrp_optimized_capacity_solution.csv'")

def main():
    """Main function with optimized capacity targeting 90%"""
    
    distance_matrix_file = './inputs/distMatrixWithDepot.csv'
    stops_data_file = './inputs/mdvrp_data.csv'
    
    try:
        print("🚌 Starting MDVRP with Optimized 90% Capacity Utilization...")
        
        # Create solver targeting 90% utilization
        solver = OptimizedCapacityMDVRP(
            distance_matrix_file=distance_matrix_file,
            stops_data_file=stops_data_file,
            depot_allocation=None  # Auto-calculate for 90% target
        )
        
        # Solve with extended time limit
        solution = solver.solve_mdvrp(time_limit_seconds=2400)  # 40 minutes
        
        if solution:
            print(f"\n🎉 SUCCESS!")
            print(f"✅ Transported {solution['total_students']} students")
            print(f"✅ Used {solution['vehicles_used']} out of {solver.total_vehicles} buses")
            print(f"✅ Average utilization: {solution['average_utilization']:.1f}%")
            print(f"✅ Maximum utilization: {solution['max_utilization']:.1f}%")
            print(f"✅ Total distance: {solution['total_distance']/1000:.1f} km")
            
            if solution['average_utilization'] <= 90:
                print(f"🎯 TARGET ACHIEVED: Average utilization ≤ 90%")
            else:
                print(f"⚠️  Utilization slightly above target")
                
        else:
            print(f"\n❌ FAILED to find solution")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()