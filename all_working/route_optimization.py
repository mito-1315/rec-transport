import pandas as pd
import numpy as np
import requests
import json
import math
from typing import List, Dict, Tuple, Optional
import folium
from folium import plugins
from sklearn.cluster import KMeans
import itertools
from geopy.distance import geodesic
from datetime import datetime
import os
import time

class OptimalBusRouteOptimizer:
    def __init__(self, google_api_key: str):
        """
        Initialize the Optimal Bus Route Optimizer with Google Routes API
        
        Args:
            google_api_key: Your Google Maps API key with Routes API enabled
        """
        self.api_key = google_api_key
        self.routes_api_url = "https://routes.googleapis.com/directions/v2:computeRoutes"
        
        # Bus constraints
        self.max_capacity = 55
        self.pickup_time_seconds = 60  # Time to pick up students at each stop
        
        # Create output directory
        self.output_dir = "optimal_route_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Rate limiting for Google API
        self.api_delay = 0.1  # 100ms delay between requests
        
    def load_data(self, bus_stops_file: str, depots_file: str, college_coords: Tuple[float, float]):
        """
        Load and validate input data
        """
        # Load bus stops
        self.bus_stops = pd.read_csv(bus_stops_file)
        
        # Load depots
        self.depots = pd.read_csv(depots_file)
        
        # College coordinates
        self.college_lat, self.college_lon = college_coords
        
        # Add unique IDs if not present
        if 'stop_id' not in self.bus_stops.columns:
            self.bus_stops['stop_id'] = range(len(self.bus_stops))
        if 'depot_id' not in self.depots.columns:
            self.depots['depot_id'] = range(len(self.depots))
        
        # Calculate total students and required buses
        self.total_students = int(self.bus_stops['num_students'].sum())
        self.required_buses = math.ceil(self.total_students / self.max_capacity)
        
        print(f"📊 Data Loaded Successfully:")
        print(f"   • Total students: {self.total_students}")
        print(f"   • Required buses: {self.required_buses}")
        print(f"   • Number of bus stops: {len(self.bus_stops)}")
        print(f"   • Number of depots: {len(self.depots)}")
        print(f"   • College location: {self.college_lat:.4f}, {self.college_lon:.4f}")
        
    def get_route_with_waypoints_optimization(self, start: Tuple[float, float], 
                                            end: Tuple[float, float], 
                                            waypoints: List[Tuple[float, float]]) -> Dict:
        """
        Get optimized route using Google Routes API with waypoint optimization
        This ensures routes follow major roads and highways
        """
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline,routes.legs,routes.optimizedIntermediateWaypointIndex"
        }
        
        request_body = {
            "origin": {
                "location": {
                    "latLng": {
                        "latitude": float(start[0]),
                        "longitude": float(start[1])
                    }
                }
            },
            "destination": {
                "location": {
                    "latLng": {
                        "latitude": float(end[0]),
                        "longitude": float(end[1])
                    }
                }
            },
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_UNAWARE",
            "optimizeWaypointOrder": True,
            "computeAlternativeRoutes": False,
            "routeModifiers": {
                "avoidTolls": False,
                "avoidHighways": False,
                "avoidFerries": True
            },
            "languageCode": "en-US",
            "units": "METRIC"
        }

        # Add waypoints
        if waypoints:
            request_body["intermediates"] = [
                {
                    "location": {
                        "latLng": {
                            "latitude": float(wp[0]),
                            "longitude": float(wp[1])
                        }
                    }
                }
                for wp in waypoints
            ]

        try:
            # Add delay to respect API rate limits
            time.sleep(self.api_delay)
            
            response = requests.post(
                self.routes_api_url,
                headers=headers,
                json=request_body,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"    ✅ Google Routes API success - Status: {response.status_code}")
                return result
            else:
                print(f"    ❌ Google Routes API Error: {response.status_code}")
                print(f"    Response: {response.text[:200]}...")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"    ❌ Route request failed: {e}")
            return None
    
    def smart_cluster_stops(self) -> Dict[int, List[int]]:
        """
        Intelligently cluster bus stops considering:
        1. Geographic proximity
        2. Capacity constraints
        3. Road network connectivity (using Google API)
        """
        print("🧠 Smart clustering bus stops...")
        
        # Use coordinates for initial clustering
        coords = self.bus_stops[['snapped_lat', 'snapped_lon']].values
        
        # Start with more clusters than needed, then merge intelligently
        initial_clusters = min(self.required_buses * 2, len(self.bus_stops))
        kmeans = KMeans(n_clusters=initial_clusters, random_state=42, n_init=10)
        initial_labels = kmeans.fit_predict(coords)
        
        # Group stops by initial cluster
        initial_cluster_dict = {}
        for i, cluster_id in enumerate(initial_labels):
            if cluster_id not in initial_cluster_dict:
                initial_cluster_dict[cluster_id] = []
            initial_cluster_dict[cluster_id].append(i)
        
        print(f"   • Initial clusters created: {len(initial_cluster_dict)}")
        
        # Intelligently merge and optimize clusters
        final_clusters = {}
        cluster_counter = 0
        
        for cluster_id, stop_indices in initial_cluster_dict.items():
            cluster_students = sum(int(self.bus_stops.iloc[idx]['num_students']) for idx in stop_indices)
            
            if cluster_students <= self.max_capacity:
                # Cluster fits in one bus
                final_clusters[cluster_counter] = stop_indices
                cluster_counter += 1
            else:
                # Need to split cluster optimally
                print(f"   • Splitting oversized cluster with {cluster_students} students")
                split_clusters = self._split_cluster_optimally(stop_indices)
                
                for split_cluster in split_clusters:
                    if split_cluster:  # Only add non-empty clusters
                        final_clusters[cluster_counter] = split_cluster
                        cluster_counter += 1
        
        print(f"   • Final optimized clusters: {len(final_clusters)}")
        return final_clusters
    
    def _split_cluster_optimally(self, stop_indices: List[int]) -> List[List[int]]:
        """
        Split an oversized cluster into multiple bus-sized clusters optimally
        """
        # Sort stops by student count (descending) for better packing
        sorted_stops = sorted(stop_indices, 
                             key=lambda x: int(self.bus_stops.iloc[x]['num_students']), 
                             reverse=True)
        
        clusters = []
        current_cluster = []
        current_capacity = 0
        
        for stop_idx in sorted_stops:
            students = int(self.bus_stops.iloc[stop_idx]['num_students'])
            
            if current_capacity + students <= self.max_capacity:
                current_cluster.append(stop_idx)
                current_capacity += students
            else:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [stop_idx]
                current_capacity = students
        
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    def find_optimal_depot_for_cluster(self, stop_indices: List[int]) -> int:
        """
        Find the optimal depot for a cluster considering:
        1. Distance from cluster centroid
        2. Road network accessibility
        """
        if not stop_indices:
            return 0
        
        # Calculate cluster centroid
        cluster_lats = [float(self.bus_stops.iloc[i]['snapped_lat']) for i in stop_indices]
        cluster_lons = [float(self.bus_stops.iloc[i]['snapped_lon']) for i in stop_indices]
        centroid = (np.mean(cluster_lats), np.mean(cluster_lons))
        
        # Find closest depots (top 3) by geodesic distance
        depot_distances = []
        for depot_idx, depot_row in self.depots.iterrows():
            depot_location = (float(depot_row['Latitude']), float(depot_row['Longitude']))
            distance = geodesic(centroid, depot_location).kilometers
            depot_distances.append((depot_idx, distance))
        
        # Sort by distance and take top 3 candidates
        depot_distances.sort(key=lambda x: x[1])
        top_depots = depot_distances[:min(3, len(depot_distances))]
        
        print(f"   • Evaluating top {len(top_depots)} depot candidates for cluster...")
        
        # Use Google Routes API to find the best depot by actual road distance
        best_depot_idx = top_depots[0][0]  # Default to closest
        best_road_distance = float('inf')
        
        for depot_idx, _ in top_depots:
            depot_location = (float(self.depots.iloc[depot_idx]['Latitude']), 
                            float(self.depots.iloc[depot_idx]['Longitude']))
            
            # Test route from centroid to depot
            route_result = self.get_route_with_waypoints_optimization(
                centroid, depot_location, []
            )
            
            if route_result and 'routes' in route_result:
                road_distance = route_result['routes'][0].get('distanceMeters', float('inf')) / 1000
                if road_distance < best_road_distance:
                    best_road_distance = road_distance
                    best_depot_idx = depot_idx
        
        return best_depot_idx
    
    def optimize_route_within_cluster(self, stop_indices: List[int], 
                                    college_location: Tuple[float, float], 
                                    depot_location: Tuple[float, float]) -> Tuple[List[int], Dict]:
        """
        Optimize route within a cluster using Google Routes API waypoint optimization
        """
        if not stop_indices:
            return [], None
        
        print(f"   • Optimizing route with {len(stop_indices)} stops using Google API...")
        
        # Create waypoints list
        waypoints = [(float(self.bus_stops.iloc[i]['snapped_lat']), 
                     float(self.bus_stops.iloc[i]['snapped_lon'])) for i in stop_indices]
        
        # Get optimized route from Google
        route_result = self.get_route_with_waypoints_optimization(
            college_location, depot_location, waypoints
        )
        
        if route_result and 'routes' in route_result:
            google_route = route_result['routes'][0]
            
            # Extract optimized waypoint order if available
            if 'optimizedIntermediateWaypointIndex' in google_route:
                optimized_indices = google_route['optimizedIntermediateWaypointIndex']
                optimized_stops = [stop_indices[i] for i in optimized_indices]
                print(f"    ✅ Google optimized waypoint order: {optimized_indices}")
            else:
                # Fallback to original order
                optimized_stops = stop_indices
                print(f"    ⚠️  Using original waypoint order")
            
            return optimized_stops, route_result
        else:
            print(f"    ❌ Google optimization failed, using nearest neighbor fallback")
            # Fallback to nearest neighbor
            optimized_stops = self.nearest_neighbor_tsp(stop_indices, college_location)
            return optimized_stops, None
    
    def nearest_neighbor_tsp(self, stop_indices: List[int], start_location: Tuple[float, float]) -> List[int]:
        """
        Fallback TSP solver using nearest neighbor heuristic
        """
        if not stop_indices:
            return []
        
        locations = [start_location]
        locations.extend([(float(self.bus_stops.iloc[i]['snapped_lat']), 
                          float(self.bus_stops.iloc[i]['snapped_lon'])) for i in stop_indices])
        
        # Calculate distance matrix
        n = len(locations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = geodesic(locations[i], locations[j]).kilometers
                matrix[i][j] = matrix[j][i] = dist
        
        # Nearest neighbor starting from start location (index 0)
        unvisited = set(range(1, len(locations)))
        route = [0]
        current = 0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: matrix[current][x])
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # Convert back to original stop indices
        return [stop_indices[i-1] for i in route[1:]]
    
    def optimize_all_routes(self):
        """
        Main optimization function - creates optimal bus routes
        """
        print("\n🚀 Starting comprehensive route optimization...")
        
        # Step 1: Smart clustering
        clusters = self.smart_cluster_stops()
        
        # Step 2: Optimize each cluster
        self.optimized_routes = []
        college_location = (float(self.college_lat), float(self.college_lon))
        
        for cluster_id, stop_indices in clusters.items():
            if not stop_indices:
                continue
            
            print(f"\n🚌 Optimizing Route {cluster_id + 1}/{len(clusters)}:")
            print(f"   • Stops in cluster: {len(stop_indices)}")
            
            # Find optimal depot
            depot_idx = self.find_optimal_depot_for_cluster(stop_indices)
            depot_location = (float(self.depots.iloc[depot_idx]['Latitude']), 
                            float(self.depots.iloc[depot_idx]['Longitude']))
            
            # Optimize route order within cluster
            optimized_order, route_details = self.optimize_route_within_cluster(
                stop_indices, college_location, depot_location
            )
            
            # Calculate route statistics
            total_students = sum(int(self.bus_stops.iloc[i]['num_students']) for i in optimized_order)
            
            # Extract route metrics from Google API
            if route_details and 'routes' in route_details:
                google_route = route_details['routes'][0]
                route_distance = google_route.get('distanceMeters', 0) / 1000  # km
                duration_str = google_route.get('duration', '0s')
                duration_seconds = int(duration_str.replace('s', ''))
                estimated_time = duration_seconds / 60  # minutes
                route_type = "Google Optimized"
                print(f"    ✅ Road-based route: {route_distance:.2f} km, {estimated_time:.1f} min")
            else:
                # Fallback calculation
                route_distance = self._calculate_fallback_distance(
                    college_location, optimized_order, depot_location
                )
                estimated_time = self._estimate_time(len(optimized_order), route_distance)
                route_type = "Estimated"
                print(f"    📏 Fallback route: {route_distance:.2f} km, {estimated_time:.1f} min")
            
            # Create route info
            route_info = {
                'bus_id': f"BUS_{cluster_id + 1:02d}",
                'cluster_id': int(cluster_id),
                'stop_indices': [int(idx) for idx in optimized_order],
                'depot_idx': int(depot_idx),
                'depot_name': str(self.depots.iloc[depot_idx]['Parking Name']),
                'depot_location': (float(self.depots.iloc[depot_idx]['Latitude']), 
                                 float(self.depots.iloc[depot_idx]['Longitude'])),
                'total_students': int(total_students),
                'capacity_utilization': float((total_students / self.max_capacity) * 100),
                'number_of_stops': int(len(optimized_order)),
                'route_details': route_details,
                'start_location': college_location,
                'estimated_distance_km': float(route_distance),
                'estimated_time_min': float(estimated_time),
                'route_type': route_type,
                'optimization_timestamp': datetime.now().isoformat()
            }
            
            self.optimized_routes.append(route_info)
            print(f"    ✅ Route {route_info['bus_id']} completed: {total_students} students, {route_distance:.1f}km")
        
        print(f"\n🎉 All routes optimized! Total routes created: {len(self.optimized_routes)}")
        return self.optimized_routes
    
    def _calculate_fallback_distance(self, start: Tuple[float, float], 
                                   stop_indices: List[int], 
                                   end: Tuple[float, float]) -> float:
        """Calculate fallback distance using geodesic"""
        total_distance = 0
        current_loc = start
        
        for stop_idx in stop_indices:
            stop = self.bus_stops.iloc[stop_idx]
            next_loc = (float(stop['snapped_lat']), float(stop['snapped_lon']))
            total_distance += geodesic(current_loc, next_loc).kilometers
            current_loc = next_loc
        
        total_distance += geodesic(current_loc, end).kilometers
        return total_distance
    
    def _estimate_time(self, num_stops: int, distance_km: float) -> float:
        """Estimate travel time"""
        driving_time = (distance_km / 30) * 60  # 30 km/h average in city
        pickup_time = (num_stops * self.pickup_time_seconds) / 60
        return driving_time + pickup_time
    
    def create_optimization_summary(self):
        """Create a comprehensive optimization summary"""
        if not hasattr(self, 'optimized_routes') or not self.optimized_routes:
            print("No optimization results available")
            return
        
        print("\n" + "="*80)
        print("🚌 OPTIMAL BUS ROUTE OPTIMIZATION RESULTS 🚌".center(80))
        print("="*80)
        
        # Summary statistics
        total_distance = sum(r['estimated_distance_km'] for r in self.optimized_routes)
        total_students_served = sum(r['total_students'] for r in self.optimized_routes)
        avg_utilization = sum(r['capacity_utilization'] for r in self.optimized_routes) / len(self.optimized_routes)
        google_routes = sum(1 for r in self.optimized_routes if r['route_type'] == "Google Optimized")
        
        print(f"📊 OPTIMIZATION SUMMARY:")
        print(f"   • Total Routes Generated: {len(self.optimized_routes)} buses")
        print(f"   • Total Students Served: {total_students_served}/{self.total_students}")
        print(f"   • Total Network Distance: {total_distance:.2f} km")
        print(f"   • Average Capacity Utilization: {avg_utilization:.1f}%")
        print(f"   • Google API Optimized Routes: {google_routes}/{len(self.optimized_routes)}")
        print(f"   • Average Distance per Route: {total_distance/len(self.optimized_routes):.2f} km")
        
        # Detailed route breakdown
        print(f"\n📋 DETAILED ROUTE BREAKDOWN:")
        print("-" * 100)
        print(f"{'Bus ID':<8} {'Students':<12} {'Util%':<8} {'Stops':<6} {'Distance':<12} {'Time':<10} {'Type':<15} {'Depot'}")
        print("-" * 100)
        
        for route in self.optimized_routes:
            route_type_short = "🚗 Google" if route['route_type'] == "Google Optimized" else "📏 Est."
            print(f"{route['bus_id']:<8} "
                f"{route['total_students']}/{self.max_capacity:<7} "
                f"{route['capacity_utilization']:.1f}%{'':<4} "
                f"{route['number_of_stops']:<6} "
                f"{route['estimated_distance_km']:.2f} km{'':<6} "
                f"{route['estimated_time_min']:.1f} min{'':<4} "
                f"{route_type_short:<15} "
                f"{route['depot_name'][:20]}{'...' if len(route['depot_name']) > 20 else ''}")
        
        # Efficiency analysis
        high_eff = [r for r in self.optimized_routes if r['capacity_utilization'] > 90]
        medium_eff = [r for r in self.optimized_routes if 75 <= r['capacity_utilization'] <= 90]
        low_eff = [r for r in self.optimized_routes if r['capacity_utilization'] < 75]
        
        print(f"\n🎯 EFFICIENCY ANALYSIS:")
        print(f"   • 🟢 High Efficiency (>90%): {len(high_eff)} routes")
        print(f"   • 🟡 Medium Efficiency (75-90%): {len(medium_eff)} routes")  
        print(f"   • 🔴 Low Efficiency (<75%): {len(low_eff)} routes")
        
        print("="*80)
    
    def export_optimized_routes(self):
        """Export comprehensive route data to CSV files"""
        if not hasattr(self, 'optimized_routes') or not self.optimized_routes:
            print("No optimization results to export")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Route summary
        route_data = []
        for route in self.optimized_routes:
            route_data.append({
                'Bus_ID': route['bus_id'],
                'Total_Students': route['total_students'],
                'Capacity_Utilization_%': round(route['capacity_utilization'], 2),
                'Number_of_Stops': route['number_of_stops'],
                'Distance_KM': round(route['estimated_distance_km'], 2),
                'Estimated_Time_Minutes': round(route['estimated_time_min'], 2),
                'Route_Type': route['route_type'],
                'Depot_Name': route['depot_name'],
                'Start_Lat': route['start_location'][0],
                'Start_Lon': route['start_location'][1],
                'End_Lat': route['depot_location'][0],
                'End_Lon': route['depot_location'][1]
            })
        
        route_df = pd.DataFrame(route_data)
        route_file = f"{self.output_dir}/optimized_routes_summary_{timestamp}.csv"
        route_df.to_csv(route_file, index=False)
        
        # Detailed stop sequences
        stop_data = []
        for route in self.optimized_routes:
            for seq, stop_idx in enumerate(route['stop_indices'], 1):
                stop = self.bus_stops.iloc[stop_idx]
                stop_data.append({
                    'Bus_ID': route['bus_id'],
                    'Sequence': seq,
                    'Stop_Name': stop.get('route_name', f'Stop_{stop_idx}'),
                    'Latitude': float(stop['snapped_lat']),
                    'Longitude': float(stop['snapped_lon']),
                    'Students': int(stop['num_students'])
                })
        
        stop_df = pd.DataFrame(stop_data)
        stop_file = f"{self.output_dir}/optimized_stop_sequences_{timestamp}.csv"
        stop_df.to_csv(stop_file, index=False)
        
        print(f"\n✅ Routes exported successfully:")
        print(f"   • Route Summary: {route_file}")
        print(f"   • Stop Sequences: {stop_file}")
        
        return route_file, stop_file
    
    def create_interactive_map(self, output_file: str = None):
        """Create an interactive map showing all optimized routes, with filter per route"""
        if not hasattr(self, 'optimized_routes') or not self.optimized_routes:
            print("No optimization results available for mapping")
            return
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.output_dir}/optimal_routes_map_{timestamp}.html"
        
        # Base map
        m = folium.Map(
            location=[float(self.college_lat), float(self.college_lon)],
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        # College marker (always visible)
        folium.Marker(
            [float(self.college_lat), float(self.college_lon)],
            popup=f"<b>🎓 College</b><br>Total Students: {self.total_students}<br>Total Routes: {len(self.optimized_routes)}",
            icon=folium.Icon(color='red', icon='graduation-cap', prefix='fa')
        ).add_to(m)
        
        colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
                'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink']
        
        for i, route in enumerate(self.optimized_routes):
            color = colors[i % len(colors)]
            group_name = f"{route['bus_id']} - {route['depot_name']}"
            fg = folium.FeatureGroup(name=group_name)
            
            # Depot marker
            folium.Marker(
                [route['depot_location'][0], route['depot_location'][1]],
                popup=f"<b>🚌 {route['bus_id']} Depot</b><br>{route['depot_name']}<br>Students: {route['total_students']}",
                icon=folium.Icon(color='darkblue', icon='bus', prefix='fa')
            ).add_to(fg)
            
            # Stop markers
            for j, stop_idx in enumerate(route['stop_indices']):
                stop = self.bus_stops.iloc[stop_idx]
                folium.CircleMarker(
                    [float(stop['snapped_lat']), float(stop['snapped_lon'])],
                    radius=max(5, int(stop['num_students']) // 2),
                    popup=f"<b>{route['bus_id']} - Stop {j+1}</b><br>Students: {int(stop['num_students'])}",
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(fg)
            
            # Route path
            if route['route_details'] and 'routes' in route['route_details']:
                try:
                    encoded_polyline = route['route_details']['routes'][0]['polyline']['encodedPolyline']
                    decoded_points = self._decode_polyline(encoded_polyline)
                    if decoded_points:
                        folium.PolyLine(
                            decoded_points,
                            color=color,
                            weight=4,
                            opacity=0.8,
                            popup=f"{route['bus_id']}: {route['estimated_distance_km']:.1f}km"
                        ).add_to(fg)
                except:
                    pass
            
            # Add the FeatureGroup for this route
            fg.add_to(m)
        
        # Layer control and plugins
        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        plugins.MeasureControl().add_to(m)
        
        m.save(output_file)
        print(f"📍 Interactive map saved: {output_file}")
        return m
    
    def _decode_polyline(self, encoded_string):
        """Decode Google polyline encoding"""
        coordinates = []
        index = 0
        lat = 0
        lng = 0
        
        while index < len(encoded_string):
            # Decode latitude
            b, shift, result = 0, 0, 0
            while True:
                b = ord(encoded_string[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            dlat = ~(result >> 1) if result & 1 else result >> 1
            lat += dlat
            
            # Decode longitude
            shift, result = 0, 0
            while True:
                b = ord(encoded_string[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            dlng = ~(result >> 1) if result & 1 else result >> 1
            lng += dlng
            
            coordinates.append([lat / 1e5, lng / 1e5])
        
        return coordinates

# Example usage and testing
def main():
    """
    Example usage of the Optimal Bus Route Optimizer
    """
    import dotenv
    import os
    dotenv.load_dotenv()

    # Initialize optimizer with your Google API key
    API_KEY = os.getenv("GOOGLE_MAP_API_KEY")  # Replace with your actual API key
    
    optimizer = OptimalBusRouteOptimizer(API_KEY)
    
    # Load your data files
    # Make sure your CSV files have the required columns:
    # Bus stops CSV: 'snapped_lat', 'snapped_lon', 'num_students', 'route_name' (optional)
    # Depots CSV: 'Latitude', 'Longitude', 'Parking Name'
    
    college_coordinates = (13.008658975353494, 80.00348150941481)  # Replace with your college coordinates (lat, lon)
    
    try:
        # Load data
        optimizer.load_data(
            bus_stops_file="Routes_Data/Friday/3_pm/3_pm_centroids_snapped.csv",
            depots_file="Routes_Data/Bus_Stops/coords.csv", 
            college_coords=college_coordinates
        )
        
        # Run optimization
        print("\n🚀 Starting route optimization...")
        optimized_routes = optimizer.optimize_all_routes()
        
        # Display results
        optimizer.create_optimization_summary()
        
        # Export results
        optimizer.export_optimized_routes()
        
        # Create interactive map
        optimizer.create_interactive_map()
        
        print("\n✅ Optimization complete! Check the output directory for results.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required files. {e}")
        print("Make sure you have:")
        print("  1. bus_stops.csv with columns: snapped_lat, snapped_lon, num_students")
        print("  2. depots.csv with columns: Latitude, Longitude, Parking Name")
    except Exception as e:
        print(f"❌ Error during optimization: {e}")

class RouteAnalyzer:
    """
    Additional utility class for analyzing and validating routes
    """
    def __init__(self, optimizer: OptimalBusRouteOptimizer):
        self.optimizer = optimizer
    
    def validate_routes(self):
        """Validate that all routes meet constraints"""
        if not hasattr(self.optimizer, 'optimized_routes'):
            print("No routes to validate")
            return False
        
        print("\n🔍 Validating optimized routes...")
        
        all_valid = True
        total_students_check = 0
        
        for route in self.optimizer.optimized_routes:
            bus_id = route['bus_id']
            students = route['total_students']
            
            # Check capacity constraint
            if students > self.optimizer.max_capacity:
                print(f"❌ {bus_id}: Exceeds capacity ({students} > {self.optimizer.max_capacity})")
                all_valid = False
            
            # Check minimum efficiency
            if route['capacity_utilization'] < 50:
                print(f"⚠️  {bus_id}: Low efficiency ({route['capacity_utilization']:.1f}%)")
            
            total_students_check += students
        
        # Check total students served
        if total_students_check != self.optimizer.total_students:
            print(f"❌ Student count mismatch: {total_students_check} vs {self.optimizer.total_students}")
            all_valid = False
        
        if all_valid:
            print("✅ All routes pass validation!")
        
        return all_valid
    
    def compare_with_baseline(self):
        """Compare optimized routes with a simple baseline"""
        if not hasattr(self.optimizer, 'optimized_routes'):
            print("No optimized routes to compare")
            return
        
        print("\n📊 Comparing with baseline (simple clustering)...")
        
        # Simple baseline: divide stops geographically without optimization
        baseline_distance = self._calculate_baseline_distance()
        optimized_distance = sum(r['estimated_distance_km'] for r in self.optimizer.optimized_routes)
        
        improvement = ((baseline_distance - optimized_distance) / baseline_distance) * 100
        
        print(f"   • Baseline total distance: {baseline_distance:.2f} km")
        print(f"   • Optimized total distance: {optimized_distance:.2f} km")
        print(f"   • Improvement: {improvement:.1f}% reduction")
        
        return improvement
    
    def _calculate_baseline_distance(self):
        """Calculate baseline distance using simple geographic clustering"""
        # Simple K-means without optimization
        from sklearn.cluster import KMeans
        
        coords = self.optimizer.bus_stops[['snapped_lat', 'snapped_lon']].values
        kmeans = KMeans(n_clusters=self.optimizer.required_buses, random_state=42)
        clusters = kmeans.fit_predict(coords)
        
        total_distance = 0
        college_location = (self.optimizer.college_lat, self.optimizer.college_lon)
        
        for cluster_id in range(self.optimizer.required_buses):
            cluster_stops = [i for i, c in enumerate(clusters) if c == cluster_id]
            if not cluster_stops:
                continue
            
            # Simple path: college -> stops in order -> nearest depot
            current_location = college_location
            
            for stop_idx in cluster_stops:
                stop = self.optimizer.bus_stops.iloc[stop_idx]
                stop_location = (float(stop['snapped_lat']), float(stop['snapped_lon']))
                total_distance += geodesic(current_location, stop_location).kilometers
                current_location = stop_location
            
            # To nearest depot
            nearest_depot_dist = min(
                geodesic(current_location, 
                        (float(self.optimizer.depots.iloc[i]['Latitude']), 
                         float(self.optimizer.depots.iloc[i]['Longitude']))).kilometers
                for i in range(len(self.optimizer.depots))
            )
            total_distance += nearest_depot_dist
        
        return total_distance
    
    def generate_detailed_report(self, output_file: str = None):
        """Generate a detailed analysis report"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.optimizer.output_dir}/detailed_analysis_report_{timestamp}.txt"
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DETAILED BUS ROUTE OPTIMIZATION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            total_distance = sum(r['estimated_distance_km'] for r in self.optimizer.optimized_routes)
            total_time = sum(r['estimated_time_min'] for r in self.optimizer.optimized_routes)
            avg_utilization = sum(r['capacity_utilization'] for r in self.optimizer.optimized_routes) / len(self.optimizer.optimized_routes)
            
            f.write("OPTIMIZATION SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Students: {self.optimizer.total_students}\n")
            f.write(f"Total Routes: {len(self.optimizer.optimized_routes)}\n")
            f.write(f"Total Distance: {total_distance:.2f} km\n")
            f.write(f"Total Time: {total_time:.1f} minutes\n")
            f.write(f"Average Capacity Utilization: {avg_utilization:.1f}%\n")
            f.write(f"Average Distance per Route: {total_distance/len(self.optimizer.optimized_routes):.2f} km\n\n")
            
            # Route details
            f.write("DETAILED ROUTE BREAKDOWN:\n")
            f.write("-" * 40 + "\n")
            for route in self.optimizer.optimized_routes:
                f.write(f"\n{route['bus_id']}:\n")
                f.write(f"  Students: {route['total_students']}/{self.optimizer.max_capacity} ({route['capacity_utilization']:.1f}%)\n")
                f.write(f"  Stops: {route['number_of_stops']}\n")
                f.write(f"  Distance: {route['estimated_distance_km']:.2f} km\n")
                f.write(f"  Time: {route['estimated_time_min']:.1f} minutes\n")
                f.write(f"  Depot: {route['depot_name']}\n")
                f.write(f"  Route Type: {route['route_type']}\n")
                
                # Stop sequence
                f.write(f"  Stop Sequence:\n")
                for i, stop_idx in enumerate(route['stop_indices']):
                    stop = self.optimizer.bus_stops.iloc[stop_idx]
                    f.write(f"    {i+1}. {stop.get('route_name', f'Stop_{stop_idx}')} ({int(stop['num_students'])} students)\n")
            
            # Efficiency analysis
            f.write("\nEFFICIENCY ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            high_eff = [r for r in self.optimizer.optimized_routes if r['capacity_utilization'] > 90]
            medium_eff = [r for r in self.optimizer.optimized_routes if 75 <= r['capacity_utilization'] <= 90]
            low_eff = [r for r in self.optimizer.optimized_routes if r['capacity_utilization'] < 75]
            
            f.write(f"High Efficiency Routes (>90%): {len(high_eff)}\n")
            f.write(f"Medium Efficiency Routes (75-90%): {len(medium_eff)}\n")
            f.write(f"Low Efficiency Routes (<75%): {len(low_eff)}\n")
            
            if low_eff:
                f.write("\nLow efficiency routes for review:\n")
                for route in low_eff:
                    f.write(f"  {route['bus_id']}: {route['capacity_utilization']:.1f}% utilization\n")
        
        print(f"📄 Detailed report saved: {output_file}")
        return output_file

# Enhanced example usage
def run_complete_optimization():
    """
    Complete optimization workflow with validation and analysis
    """
    # Configuration
    import dotenv
    import os
    dotenv.load_dotenv()


    API_KEY = os.getenv("GOOGLE_MAP_API_KEY")
    COLLEGE_COORDS = (13.008658975353494, 80.00348150941481)  # Bangalore coordinates example
    
    print("🚌 Starting Complete Bus Route Optimization System")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = OptimalBusRouteOptimizer(API_KEY)
    
    try:
        # Step 1: Load data
        bus_stops_file = "Routes_Data/Friday/3_pm/3_pm_centroids_snapped.csv"
        depots_file = "Routes_Data/Bus_Stops/coords.csv"
        print("\n📂 Step 1: Loading data...")
        optimizer.load_data(bus_stops_file, depots_file, COLLEGE_COORDS)
        
        # Step 2: Run optimization
        print("\n🔧 Step 2: Running optimization...")
        routes = optimizer.optimize_all_routes()
        
        # Step 3: Display results
        print("\n📊 Step 3: Analyzing results...")
        optimizer.create_optimization_summary()
        
        # Step 4: Validate routes
        print("\n✅ Step 4: Validating routes...")
        analyzer = RouteAnalyzer(optimizer)
        analyzer.validate_routes()
        
        # Step 5: Compare with baseline
        print("\n📈 Step 5: Performance comparison...")
        improvement = analyzer.compare_with_baseline()
        
        # Step 6: Export results
        print("\n💾 Step 6: Exporting results...")
        optimizer.export_optimized_routes()
        optimizer.create_interactive_map()
        analyzer.generate_detailed_report()
        
        print("\n🎉 Optimization completed successfully!")
        print(f"📊 Summary: {len(routes)} routes created with {improvement:.1f}% distance improvement")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure your Google API key has Routes API enabled")
        print("2. Check that CSV files exist and have required columns")
        print("3. Verify college coordinates are correct")
        print("4. Check internet connection for Google API calls")

if __name__ == "__main__":
    # You can run either the simple main() or the complete optimization
    # main()
    run_complete_optimization()