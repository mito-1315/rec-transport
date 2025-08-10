import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import seaborn as sns

class AllBusesMDVRP:
    def __init__(self, distance_matrix_file, stops_data_file, depot_allocation=None):
        """Initialize MDVRP solver using ALL available buses"""
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
        """Setup MDVRP parameters using ALL available buses"""
        print(f"\n{'='*60}")
        print("USING ALL AVAILABLE BUSES FROM DEPOTS")
        print("="*60)
        
        total_students = self.stops_df['num_students'].sum()
        
        # **USE ALL BUSES FROM YOUR ORIGINAL ALLOCATION**
        if depot_allocation is None:
            # Use the EXACT bus allocation from your coords.csv
            self.depot_bus_allocation = {
                0: 2,   # Mudichur
                1: 1,   # Urapakkam (Vandalur)
                2: 3,   # KPM (White gate)
                3: 1,   # Cheyyar
                4: 2,   # Arakkonam
                5: 9,   # Andarkuppam
                6: 5,   # Ayapakkam (Giri Parking)
                7: 14,  # Kovilambakkam (13+1W = 14)
                8: 22,  # Vyasarpadi (20+2W = 22)
                9: 3,   # Red Hills
                10: 11, # Velachery (10+W = 11, assuming W=1)
                11: 4,  # Avadi (Kavarapalayam)
                12: 4,  # Veppamapattu
                13: 4,  # Ennore
                14: 2,  # VGP (Golden Beach)
                15: 2,  # Chengalpet
                16: 1,  # Minjur
                17: 3,  # Dusi (Sekar)
                18: 18, # Lucas
                19: 16, # Sathya Garden
                20: 3,  # Veerapuram
                21: 1,  # Thiruvallur (Manavalan nagar)
                22: 6,  # Kallikuppam
                23: 1,  # Thiruvallur (Venakatesan G)
            }
        else:
            self.depot_bus_allocation = depot_allocation
            
        # Calculate total buses
        total_buses = sum(self.depot_bus_allocation.values())
        
        print(f"🚌 USING ALL AVAILABLE BUSES:")
        print(f"- Total buses from all depots: {total_buses}")
        
        for depot_idx, buses in self.depot_bus_allocation.items():
            if depot_idx < len(self.stops_df):
                depot_name = self.stops_df.iloc[depot_idx]['stop_name']
                print(f"  Depot {depot_idx} ({depot_name}): {buses} buses")
        
        self.vehicle_capacity = 55  # Standard bus capacity
        self.total_vehicles = total_buses
        
        # Calculate capacity utilization
        total_capacity = self.total_vehicles * self.vehicle_capacity
        utilization = (total_students / total_capacity) * 100
        
        print(f"\n📊 CAPACITY ANALYSIS:")
        print(f"- Total vehicles: {self.total_vehicles}")
        print(f"- Vehicle capacity: {self.vehicle_capacity} students each")
        print(f"- Total theoretical capacity: {total_capacity} students")
        print(f"- Total students to transport: {total_students}")
        print(f"- Overall capacity utilization: {utilization:.1f}%")
        print(f"- Average students per bus: {total_students / self.total_vehicles:.1f}")
        
        if utilization < 50:
            print(f"✅ Excellent! Very low utilization - plenty of capacity")
        elif utilization < 70:
            print(f"✅ Good! Comfortable utilization level")
        elif utilization < 85:
            print(f"✅ Acceptable utilization level")
        else:
            print(f"⚠️  High utilization - buses will be quite full")
        
        # Create vehicle start and end points for ALL buses
        self.vehicle_starts = []
        self.vehicle_ends = []
        
        school_index = self.school_indices[0] if self.school_indices else len(self.stops_df) - 1
        
        vehicle_id = 0
        for depot_idx, num_buses in self.depot_bus_allocation.items():
            for bus_num in range(num_buses):
                self.vehicle_starts.append(depot_idx)
                self.vehicle_ends.append(school_index)
                vehicle_id += 1
        
        print(f"\n🚌 VEHICLE CONFIGURATION:")
        print(f"- Total vehicles configured: {len(self.vehicle_starts)}")
        print(f"- All vehicles start from their assigned depots")
        print(f"- All vehicles end at school (index {school_index})")
        print(f"- Vehicle starts: {self.vehicle_starts[:10]}... (showing first 10)")
        
    def solve_mdvrp(self, time_limit_seconds=7200):
        """Solve the MDVRP problem using ALL buses"""
        print(f"\n{'='*60}")
        print("SOLVING MDVRP WITH ALL AVAILABLE BUSES")
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
        print(f"- Total vehicles: {data['num_vehicles']}")
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
        
        # **CAPACITY CONSTRAINT** 
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        # Use full capacity
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # No slack needed with so many buses
            data['vehicle_capacities'],  # Full capacity allowed
            True,
            'Capacity'
        )
        
        # **FIXED SEARCH PARAMETERS** - Removed invalid parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        
        # Use simpler but effective strategies for large problems
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(time_limit_seconds)
        search_parameters.log_search = True
        
        print(f"🚀 Starting optimization with {self.total_vehicles} vehicles...")
        print(f"⏱️  Using PARALLEL_CHEAPEST_INSERTION + GUIDED_LOCAL_SEARCH")
        print(f"⏱️  This may take up to {time_limit_seconds/60:.0f} minutes...")
        
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            print(f"\n✅ SOLUTION FOUND!")
            return self.process_solution(data, manager, routing, solution)
        else:
            print(f"❌ No solution found after {time_limit_seconds} seconds")
            print("\n🔧 Let's try with a simpler approach...")
            return self.try_fallback_solution(data, manager, routing, time_limit_seconds)
            
    def try_fallback_solution(self, data, manager, routing, time_limit_seconds):
        """Try a fallback approach with different parameters"""
        print(f"\n🔄 TRYING FALLBACK APPROACH...")
        
        # Try with different strategy
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
        )
        search_parameters.time_limit.FromSeconds(time_limit_seconds // 2)  # Half time
        search_parameters.log_search = True
        
        print(f"Trying PATH_CHEAPEST_ARC strategy...")
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            print(f"\n✅ FALLBACK SOLUTION FOUND!")
            return self.process_solution(data, manager, routing, solution)
        else:
            print(f"❌ Fallback also failed")
            print("\n💡 Suggestions:")
            print("1. Problem might be too large for current OR-Tools version")
            print("2. Try reducing number of vehicles")
            print("3. Increase time limit to 2-4 hours")
            print("4. Consider solving in batches")
            return None
            
    def process_solution(self, data, manager, routing, solution):
        """Process and display solution with detailed bus usage"""
        print(f"\n{'='*50}")
        print("✅ SOLUTION FOUND WITH ALL BUSES!")
        print(f"{'='*50}")
        print(f"Objective (Total Distance): {solution.ObjectiveValue():,} meters")
        print(f"Objective (Total Distance): {solution.ObjectiveValue()/1000:.1f} km")
        
        routes = []
        total_distance = 0
        total_students = 0
        depot_usage = {depot_idx: {'allocated': buses, 'used': 0} 
                      for depot_idx, buses in self.depot_bus_allocation.items()}
        
        active_buses = 0
        empty_buses = 0
        
        print(f"\n🚌 Processing {data['num_vehicles']} vehicle routes...")
        
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
            
            start_depot_idx = route_stops[0]
            depot_usage[start_depot_idx]['used'] += 1
            
            if len(route_stops) > 2:  # Active routes
                active_buses += 1
                
                route_info = {
                    'vehicle_id': vehicle_id,
                    'stops': route_stops,
                    'distance_m': route_distance,
                    'distance_km': route_distance / 1000,
                    'students': route_students,
                    'capacity_utilization': (route_students / self.vehicle_capacity) * 100,
                    'start_depot': start_depot_idx,
                    'end_school': route_stops[-1],
                    'num_pickup_stops': len(route_stops) - 2
                }
                routes.append(route_info)
                
                total_distance += route_distance
                total_students += route_students
                
                # Print every 10th active route
                if active_buses % 10 == 1 or route_students > 40:
                    depot_name = self.stops_df.iloc[start_depot_idx]['stop_name']
                    print(f"Bus {vehicle_id:3d}: {route_students:2d} students, {route_distance/1000:.1f}km ({depot_name})")
                    
            else:  # Empty routes (depot -> school only)
                empty_buses += 1
        
        # **COMPREHENSIVE SUMMARY**
        print(f"\n{'='*60}")
        print("🚌 COMPLETE BUS UTILIZATION SUMMARY")
        print(f"{'='*60}")
        
        print(f"📊 Overall Statistics:")
        print(f"- Total buses available: {self.total_vehicles}")
        print(f"- Active buses (with students): {active_buses}")
        print(f"- Empty buses (depot→school only): {empty_buses}")
        print(f"- Bus utilization rate: {active_buses/self.total_vehicles*100:.1f}%")
        print(f"- Total distance: {total_distance:,} meters ({total_distance/1000:.1f} km)")
        print(f"- Total students transported: {total_students}")
        
        if active_buses > 0:
            print(f"- Average students per active bus: {total_students/active_buses:.1f}")
            print(f"- Average distance per active bus: {total_distance/active_buses:,.0f} meters")
            avg_utilization = (total_students / (active_buses * self.vehicle_capacity)) * 100
            print(f"- Average capacity utilization: {avg_utilization:.1f}%")
        
        print(f"\n🏢 Depot-wise Bus Usage:")
        for depot_idx, usage in depot_usage.items():
            if depot_idx < len(self.stops_df):
                depot_name = self.stops_df.iloc[depot_idx]['stop_name']
                allocated = usage['allocated']
                used = usage['used']
                efficiency = (used/allocated*100) if allocated > 0 else 0
                print(f"- {depot_name:25s}: {used:2d}/{allocated:2d} buses ({efficiency:3.0f}%)")
        
        # Utilization distribution
        if routes:
            utilizations = [r['capacity_utilization'] for r in routes]
            print(f"\n📈 Capacity Utilization Distribution:")
            print(f"- Range: {min(utilizations):.1f}% - {max(utilizations):.1f}%")
            
            under_25 = sum(1 for u in utilizations if u < 25)
            between_25_50 = sum(1 for u in utilizations if 25 <= u < 50)
            between_50_75 = sum(1 for u in utilizations if 50 <= u < 75)
            over_75 = sum(1 for u in utilizations if u >= 75)
            
            print(f"- Distribution:")
            print(f"  • Under 25%: {under_25:2d} buses")
            print(f"  • 25-50%:    {between_25_50:2d} buses")
            print(f"  • 50-75%:    {between_50_75:2d} buses")
            print(f"  • Over 75%:  {over_75:2d} buses")
        
        # Save solution
        self.save_solution(routes, depot_usage)
        
        return {
            'routes': routes,
            'total_distance': total_distance,
            'total_students': total_students,
            'vehicles_used': active_buses,
            'vehicles_available': self.total_vehicles,
            'empty_vehicles': empty_buses,
            'depot_usage': depot_usage,
            'objective_value': solution.ObjectiveValue()
        }
        
    def save_solution(self, routes, depot_usage):
        """Save detailed solution with all bus information"""
        # Route details
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
        solution_df.to_csv('mdvrp_all_buses_solution.csv', index=False)
        
        # Depot usage summary
        depot_summary = []
        for depot_idx, usage in depot_usage.items():
            if depot_idx < len(self.stops_df):
                depot_info = self.stops_df.iloc[depot_idx]
                depot_summary.append({
                    'depot_index': depot_idx,
                    'depot_id': depot_info['stop_id'],
                    'depot_name': depot_info['stop_name'],
                    'buses_allocated': usage['allocated'],
                    'buses_used': usage['used'],
                    'buses_unused': usage['allocated'] - usage['used'],
                    'utilization_rate': (usage['used'] / usage['allocated']) * 100 if usage['allocated'] > 0 else 0
                })
        
        depot_df = pd.DataFrame(depot_summary)
        depot_df.to_csv('depot_usage_summary.csv', index=False)
        
        print(f"\n💾 Files saved:")
        print(f"1. mdvrp_all_buses_solution.csv - Detailed route solution")
        print(f"2. depot_usage_summary.csv - Depot bus utilization")

def main():
    """Main function using ALL available buses"""
    
    distance_matrix_file = './inputs/distMatrixWithDepot.csv'
    stops_data_file = './inputs/mdvrp_data.csv'
    
    try:
        print("🚌 Starting MDVRP with ALL AVAILABLE BUSES...")
        
        # Create solver using ALL buses
        solver = AllBusesMDVRP(
            distance_matrix_file=distance_matrix_file,
            stops_data_file=stops_data_file,
            depot_allocation=None  # Uses all buses from original allocation
        )
        
        # Solve with extended time limit for larger problem
        solution = solver.solve_mdvrp(time_limit_seconds=3600)  # 1 hour
        
        if solution:
            print(f"\n🎉 SUCCESS!")
            print(f"✅ Used {solution['vehicles_used']} out of {solution['vehicles_available']} buses")
            print(f"✅ Transported {solution['total_students']} students")
            print(f"✅ Total distance: {solution['total_distance']/1000:.1f} km")
            print(f"✅ Bus efficiency: {solution['vehicles_used']/solution['vehicles_available']*100:.1f}%")
            
        else:
            print(f"\n❌ FAILED to find solution")
            print("The problem with 138 buses is very complex!")
            print("Consider:")
            print("1. Running overnight (6-12 hours)")
            print("2. Using fewer depots/buses")
            print("3. Solving in smaller batches")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()