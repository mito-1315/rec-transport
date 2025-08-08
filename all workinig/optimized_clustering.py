import pandas as pd
import numpy as np
import googlemaps
import json
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler
import folium
from folium import plugins
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from collections import defaultdict

class OptimizedBoardingPointAssignment:
    def __init__(self, student_data_path, bus_stops_path, google_maps_api_key, max_distance_km=1.0, cache_file='distance_cache.pkl'):
        """
        Initialize the optimized boarding point assignment system with minimal API calls
        
        Args:
            student_data_path: Path to student locations CSV
            bus_stops_path: Path to bus stops CSV
            google_maps_api_key: Google Maps API key
            max_distance_km: Maximum road distance in km (default 1.0km)
            cache_file: Path to persistent cache file
        """
        self.max_distance_km = max_distance_km
        self.student_data = pd.read_csv(student_data_path)
        self.bus_stops_data = pd.read_csv(bus_stops_path)
        self.cache_file = cache_file
        
        # Initialize Google Maps client
        self.gmaps = googlemaps.Client(key=google_maps_api_key)
        
        # Load persistent cache
        self.distance_cache = self.load_cache()
        
        # API call tracking
        self.api_calls_made = 0
        self.cache_hits = 0
        
        # Pre-compute straight-line distances for all pairs to minimize API calls
        self.straight_line_distances = {}
        self._precompute_straight_line_distances()
        
        print(f"Loaded {len(self.student_data)} student locations")
        print(f"Loaded {len(self.bus_stops_data)} bus stops")
        print(f"Maximum distance constraint: {self.max_distance_km} km")
        print(f"Loaded {len(self.distance_cache)} cached distances")
        
    def load_cache(self):
        """Load persistent distance cache"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                print("Warning: Could not load cache file, starting fresh")
        return {}
    
    def save_cache(self):
        """Save distance cache to file"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.distance_cache, f)
            print(f"Cache saved with {len(self.distance_cache)} entries")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _precompute_straight_line_distances(self):
        """Pre-compute all straight-line distances to avoid repeated calculations"""
        print("Pre-computing straight-line distances...")
        
        # Student coordinates
        student_coords = [(float(row['latitude']), float(row['longitude'])) 
                         for _, row in self.student_data.iterrows()]
        
        # Bus stop coordinates
        bus_stop_coords = [(float(row['latitude']), float(row['longitude'])) 
                          for _, row in self.bus_stops_data.iterrows()]
        
        # Pre-compute student to bus stop distances
        for i, student_coord in enumerate(student_coords):
            for j, stop_coord in enumerate(bus_stop_coords):
                key = f"student_{i}_stop_{j}"
                self.straight_line_distances[key] = geodesic(student_coord, stop_coord).kilometers
        
        # Pre-compute student to student distances (for clustering analysis)
        for i, coord1 in enumerate(student_coords):
            for j, coord2 in enumerate(student_coords[i+1:], i+1):
                key = f"student_{i}_student_{j}"
                self.straight_line_distances[key] = geodesic(coord1, coord2).kilometers
    
    def get_straight_line_distance(self, type1, idx1, type2, idx2):
        """Get pre-computed straight-line distance"""
        if type1 == type2 and idx1 == idx2:
            return 0.0
        
        # Ensure consistent ordering for cache keys
        if (type1, idx1) > (type2, idx2):
            type1, idx1, type2, idx2 = type2, idx2, type1, idx1
        
        key = f"{type1}_{idx1}_{type2}_{idx2}"
        return self.straight_line_distances.get(key, float('inf'))
    
    def create_distance_key(self, lat1, lon1, lat2, lon2):
        """Create a consistent cache key for distance pairs"""
        # Round coordinates to 6 decimal places for consistency
        coord1 = (round(lat1, 6), round(lon1, 6))
        coord2 = (round(lat2, 6), round(lon2, 6))
        
        # Ensure consistent ordering
        if coord1 > coord2:
            coord1, coord2 = coord2, coord1
        
        return f"{coord1[0]},{coord1[1]}|{coord2[0]},{coord2[1]}"
    
    def intelligent_pre_filtering(self, student_coords, bus_stop_coords, max_candidates=5):
        """
        Intelligently pre-filter bus stops using straight-line distance and geometric analysis
        Returns only the most promising candidates to minimize API calls
        """
        candidates = defaultdict(list)  # student_idx -> list of (stop_idx, straight_distance)
        
        for student_idx, student_coord in enumerate(student_coords):
            # Get all stops within 1.5x the distance limit (straight-line)
            potential_stops = []
            
            for stop_idx, stop_coord in enumerate(bus_stop_coords):
                straight_dist = self.get_straight_line_distance('student', student_idx, 'stop', stop_idx)
                
                # Only consider stops within a reasonable straight-line distance
                # Road distance is typically 1.2-1.8x straight-line distance in urban areas
                if straight_dist <= self.max_distance_km * 1.4:
                    potential_stops.append((stop_idx, straight_dist))
            
            # Sort by straight-line distance and take top candidates
            potential_stops.sort(key=lambda x: x[1])
            candidates[student_idx] = potential_stops[:max_candidates]
        
        return candidates
    
    def calculate_road_distance_ultra_efficient(self, student_coords, bus_stop_coords, pre_filtered_candidates):
        """
        Ultra-efficient road distance calculation with minimal API calls
        """
        distances = {}
        uncached_pairs = []
        batch_origins = []
        batch_destinations = []
        batch_mapping = {}  # Maps batch indices to original indices
        
        # First pass: Check cache and collect uncached pairs
        total_pairs_needed = sum(len(candidates) for candidates in pre_filtered_candidates.values())
        
        print(f"Checking cache for {total_pairs_needed} student-stop pairs...")
        
        for student_idx, candidates in pre_filtered_candidates.items():
            student_coord = student_coords[student_idx]
            
            for stop_idx, _ in candidates:
                stop_coord = bus_stop_coords[stop_idx]
                cache_key = self.create_distance_key(
                    student_coord[0], student_coord[1],
                    stop_coord[0], stop_coord[1]
                )
                
                if cache_key in self.distance_cache:
                    distances[(student_idx, stop_idx)] = self.distance_cache[cache_key]
                    self.cache_hits += 1
                else:
                    uncached_pairs.append((student_idx, stop_idx, cache_key))
        
        print(f"Cache hits: {self.cache_hits}, Need to calculate: {len(uncached_pairs)}")
        
        if not uncached_pairs:
            return distances
        
        # Optimize batching by grouping nearby coordinates
        unique_origins = {}
        unique_destinations = {}
        
        for student_idx, stop_idx, cache_key in uncached_pairs:
            student_coord = student_coords[student_idx]
            stop_coord = bus_stop_coords[stop_idx]
            
            # Use rounded coordinates to group nearby points
            origin_key = (round(student_coord[0], 4), round(student_coord[1], 4))
            dest_key = (round(stop_coord[0], 4), round(stop_coord[1], 4))
            
            if origin_key not in unique_origins:
                unique_origins[origin_key] = student_coord
            if dest_key not in unique_destinations:
                unique_destinations[dest_key] = stop_coord
        
        origin_list = list(unique_origins.values())
        dest_list = list(unique_destinations.values())
        
        print(f"Optimized to {len(origin_list)} unique origins and {len(dest_list)} unique destinations")
        
        # Calculate distances in optimized batches
        batch_results = self.calculate_distance_matrix_batch(origin_list, dest_list)
        
        # Map results back to original pairs
        for student_idx, stop_idx, cache_key in uncached_pairs:
            student_coord = student_coords[student_idx]
            stop_coord = bus_stop_coords[stop_idx]
            
            # Find the corresponding indices in the optimized batch
            origin_key = (round(student_coord[0], 4), round(student_coord[1], 4))
            dest_key = (round(stop_coord[0], 4), round(stop_coord[1], 4))
            
            try:
                orig_idx = list(unique_origins.keys()).index(origin_key)
                dest_idx = list(unique_destinations.keys()).index(dest_key)
                
                distance = batch_results.get((orig_idx, dest_idx))
                if distance is not None:
                    distances[(student_idx, stop_idx)] = distance
                    self.distance_cache[cache_key] = distance
                else:
                    # Fallback to straight-line distance
                    distances[(student_idx, stop_idx)] = self.get_straight_line_distance(
                        'student', student_idx, 'stop', stop_idx
                    )
            except (ValueError, KeyError):
                # Fallback to straight-line distance
                distances[(student_idx, stop_idx)] = self.get_straight_line_distance(
                    'student', student_idx, 'stop', stop_idx
                )
        
        return distances
    
    def calculate_distance_matrix_batch(self, origins, destinations, max_batch_size=25):
        """
        Calculate distance matrix in optimized batches with error handling
        """
        distances = {}
        
        total_batches = (len(origins) + max_batch_size - 1) // max_batch_size * (len(destinations) + max_batch_size - 1) // max_batch_size
        
        with tqdm(total=total_batches, desc="API calls") as pbar:
            for i in range(0, len(origins), max_batch_size):
                origin_batch = origins[i:i + max_batch_size]
                
                for j in range(0, len(destinations), max_batch_size):
                    dest_batch = destinations[j:j + max_batch_size]
                    
                    try:
                        matrix = self.gmaps.distance_matrix(
                            origins=origin_batch,
                            destinations=dest_batch,
                            mode="driving",
                            units="metric",
                            avoid="tolls"
                        )
                        
                        self.api_calls_made += 1
                        
                        if matrix['status'] == 'OK':
                            for orig_idx, row in enumerate(matrix['rows']):
                                for dest_idx, element in enumerate(row['elements']):
                                    global_orig_idx = i + orig_idx
                                    global_dest_idx = j + dest_idx
                                    
                                    if element['status'] == 'OK':
                                        distance_km = element['distance']['value'] / 1000.0
                                        distances[(global_orig_idx, global_dest_idx)] = distance_km
                                    else:
                                        distances[(global_orig_idx, global_dest_idx)] = None
                        
                        # Rate limiting - be more aggressive to stay within quotas
                        time.sleep(0.2)
                        
                    except Exception as e:
                        print(f"API call failed: {e}")
                        # Continue with other batches
                    
                    pbar.update(1)
        
        return distances
    
    def create_student_clusters_hdbscan(self, min_cluster_size=8, min_samples=5):
        """
        Create student clusters using HDBSCAN algorithm with pre-computed distances
        """
        print("Creating student clusters using HDBSCAN...")
        
        # Use pre-computed straight-line distances for initial clustering
        student_coords = self.student_data[['latitude', 'longitude']].astype(float).values
        student_coords_rad = np.radians(student_coords)
        
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='haversine'
        )
        
        cluster_labels = clusterer.fit_predict(student_coords_rad)
        
        # Analyze clusters
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        
        # Add cluster labels to student data
        self.student_data['cluster_id'] = cluster_labels
        
        return cluster_labels, clusterer
    
    def find_optimal_centroid_for_cluster_efficient(self, cluster_students):
        """
        Find optimal boarding point with minimal API calls using smart pre-filtering
        """
        if len(cluster_students) == 0:
            return None
        
        # Get student coordinates
        student_coords = [(float(row['latitude']), float(row['longitude'])) 
                         for _, row in cluster_students.iterrows()]
        
        # Get student indices in original dataset
        student_indices = []
        for _, student in cluster_students.iterrows():
            # Find the index of this student in the original dataset
            student_idx = self.student_data[self.student_data['user'] == student['user']].index[0]
            student_indices.append(student_idx)
        
        # Bus stop coordinates
        bus_stop_coords = [(float(row['latitude']), float(row['longitude'])) 
                          for _, row in self.bus_stops_data.iterrows()]
        
        # Use intelligent pre-filtering to reduce candidates
        candidates_per_student = self.intelligent_pre_filtering(
            student_coords, bus_stop_coords, max_candidates=3  # Only top 3 candidates per student
        )
        
        # Create a set of unique candidate stops
        all_candidate_stops = set()
        for candidates in candidates_per_student.values():
            for stop_idx, _ in candidates:
                all_candidate_stops.add(stop_idx)
        
        print(f"Reduced to {len(all_candidate_stops)} candidate stops for cluster of {len(student_coords)} students")
        
        if not all_candidate_stops:
            return None
        
        # Calculate road distances only for viable candidates
        filtered_candidates = {
            i: [(stop_idx, straight_dist) for stop_idx, straight_dist in candidates 
                if stop_idx in all_candidate_stops]
            for i, candidates in candidates_per_student.items()
        }
        
        distances = self.calculate_road_distance_ultra_efficient(
            student_coords, bus_stop_coords, filtered_candidates
        )
        
        # Evaluate each candidate stop
        best_stop = None
        best_score = -1
        
        for stop_idx in all_candidate_stops:
            reachable_students = 0
            total_distance = 0
            max_student_distance = 0
            
            for student_local_idx in range(len(student_coords)):
                distance_key = (student_local_idx, stop_idx)
                road_distance = distances.get(distance_key)
                
                if road_distance is None:
                    # Fallback to pre-computed straight-line distance
                    student_global_idx = student_indices[student_local_idx]
                    road_distance = self.get_straight_line_distance('student', student_global_idx, 'stop', stop_idx)
                
                if road_distance <= self.max_distance_km:
                    reachable_students += 1
                    total_distance += road_distance
                    max_student_distance = max(max_student_distance, road_distance)
            
            # Calculate score
            if reachable_students > 0:
                stop_data = self.bus_stops_data.iloc[stop_idx]
                reachability_score = reachable_students / len(student_coords)
                avg_distance = total_distance / reachable_students
                capacity_score = min(stop_data['student_count'] / 50.0, 1.0)
                
                score = (
                    reachability_score * 100 +
                    (1 - avg_distance / self.max_distance_km) * 20 +
                    (1 - max_student_distance / self.max_distance_km) * 10 +
                    capacity_score * 5
                )
                
                if score > best_score:
                    best_score = score
                    best_stop = {
                        'stop_id': stop_data['cluster_id'],
                        'stop_lat': float(stop_data['latitude']),
                        'stop_lon': float(stop_data['longitude']),
                        'student_count': stop_data['student_count'],
                        'reachable_students': reachable_students,
                        'total_students': len(student_coords),
                        'avg_distance': avg_distance,
                        'max_distance': max_student_distance,
                        'reachability_percentage': reachability_score * 100,
                        'score': score,
                        'stop_index': stop_idx  # Add this for efficient distance lookup
                    }
        
        return best_stop
    
    def assign_boarding_points_optimized(self, min_cluster_size=8, min_samples=5):
        """
        Ultra-efficient boarding point assignment with minimal API calls
        """
        print("Starting ultra-optimized boarding point assignment...")
        
        # Create clusters
        cluster_labels, clusterer = self.create_student_clusters_hdbscan(
            min_cluster_size, min_samples
        )
        
        assignments = []
        cluster_summary = []
        
        # Process each cluster
        unique_clusters = set(cluster_labels)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        
        print(f"Processing {len(unique_clusters)} clusters with minimal API calls...")
        
        for cluster_id in tqdm(unique_clusters, desc="Processing clusters"):
            cluster_mask = cluster_labels == cluster_id
            cluster_students = self.student_data[cluster_mask].copy()
            
            print(f"\nCluster {cluster_id}: {len(cluster_students)} students")
            
            # Find optimal boarding point
            optimal_stop = self.find_optimal_centroid_for_cluster_efficient(cluster_students)
            
            if optimal_stop:
                print(f"  Assigned to stop {optimal_stop['stop_id']}")
                print(f"  Reachability: {optimal_stop['reachable_students']}/{optimal_stop['total_students']} ({optimal_stop['reachability_percentage']:.1f}%)")
                
                # Efficiently assign distances to students in this cluster
                for _, student in cluster_students.iterrows():
                    # Find student index in original dataset
                    student_global_idx = self.student_data[self.student_data['user'] == student['user']].index[0]
                    
                    # Try to get road distance from cache/calculations, fallback to straight-line
                    road_distance = self.get_straight_line_distance(
                        'student', student_global_idx, 'stop', optimal_stop['stop_index']
                    )
                    
                    # Look for cached road distance
                    cache_key = self.create_distance_key(
                        float(student['latitude']), float(student['longitude']),
                        optimal_stop['stop_lat'], optimal_stop['stop_lon']
                    )
                    
                    if cache_key in self.distance_cache:
                        road_distance = self.distance_cache[cache_key]
                    
                    assignments.append({
                        'user': student['user'],
                        'original_bus_no': student['bus_no'] if 'bus_no' in student else 'N/A',
                        'original_boarding_point': student['boarding_point_name'] if 'boarding_point_name' in student else 'N/A',
                        'student_lat': float(student['latitude']),
                        'student_lon': float(student['longitude']),
                        'assigned_stop_id': optimal_stop['stop_id'],
                        'assigned_stop_lat': optimal_stop['stop_lat'],
                        'assigned_stop_lon': optimal_stop['stop_lon'],
                        'road_distance_km': road_distance,
                        'cluster_id': cluster_id,
                        'within_distance_limit': road_distance <= self.max_distance_km
                    })
                
                cluster_summary.append({
                    'cluster_id': cluster_id,
                    'student_count': len(cluster_students),
                    'assigned_stop_id': optimal_stop['stop_id'],
                    'reachable_students': optimal_stop['reachable_students'],
                    'reachability_percentage': optimal_stop['reachability_percentage'],
                    'avg_distance': optimal_stop['avg_distance'],
                    'max_distance': optimal_stop['max_distance']
                })
        
        # Handle noise points efficiently
        noise_mask = cluster_labels == -1
        noise_students = self.student_data[noise_mask]
        
        if len(noise_students) > 0:
            print(f"\nProcessing {len(noise_students)} noise points with minimal API calls...")
            self._process_noise_points_efficient(noise_students, assignments)
        
        # Save cache before returning
        self.save_cache()
        
        assignments_df = pd.DataFrame(assignments)
        cluster_summary_df = pd.DataFrame(cluster_summary)
        
        print(f"\n📊 API Efficiency Report:")
        print(f"   Total API calls made: {self.api_calls_made}")
        print(f"   Cache hits: {self.cache_hits}")
        print(f"   Cache hit rate: {self.cache_hits/(self.cache_hits + self.api_calls_made)*100:.1f}%")
        
        return assignments_df, cluster_summary_df
    
    def _process_noise_points_efficient(self, noise_students, assignments):
        """Process noise points (individual students) with minimal API calls"""
        for _, student in noise_students.iterrows():
            student_global_idx = self.student_data[self.student_data['user'] == student['user']].index[0]
            
            # Find closest stop using pre-computed straight-line distances
            best_stop_idx = None
            best_distance = float('inf')
            
            for stop_idx in range(len(self.bus_stops_data)):
                straight_distance = self.get_straight_line_distance('student', student_global_idx, 'stop', stop_idx)
                
                if straight_distance <= self.max_distance_km * 1.2 and straight_distance < best_distance:
                    # Check if we have cached road distance
                    stop_data = self.bus_stops_data.iloc[stop_idx]
                    cache_key = self.create_distance_key(
                        float(student['latitude']), float(student['longitude']),
                        float(stop_data['latitude']), float(stop_data['longitude'])
                    )
                    
                    road_distance = self.distance_cache.get(cache_key, straight_distance * 1.3)  # Estimate if not cached
                    
                    if road_distance <= self.max_distance_km:
                        best_distance = road_distance
                        best_stop_idx = stop_idx
            
            if best_stop_idx is not None:
                stop_data = self.bus_stops_data.iloc[best_stop_idx]
                assignments.append({
                    'user': student['user'],
                    'original_bus_no': student['bus_no'] if 'bus_no' in student else 'N/A',
                    'original_boarding_point': student['boarding_point_name'] if 'boarding_point_name' in student else 'N/A',
                    'student_lat': float(student['latitude']),
                    'student_lon': float(student['longitude']),
                    'assigned_stop_id': stop_data['cluster_id'],
                    'assigned_stop_lat': float(stop_data['latitude']),
                    'assigned_stop_lon': float(stop_data['longitude']),
                    'road_distance_km': best_distance,
                    'cluster_id': -1,
                    'within_distance_limit': True
                })
    
    def create_enhanced_visualization(self, assignments_df, cluster_summary_df, output_path="optimized_boarding_assignments_map.html"):
        """Create enhanced visualization (unchanged from original)"""
        if len(assignments_df) == 0:
            print("No assignments to visualize")
            return
        
        # Calculate map center
        center_lat = assignments_df['student_lat'].mean()
        center_lon = assignments_df['student_lon'].mean()
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        # Color palette for clusters
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
                 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white',
                 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
        
        # Add bus stops with enhanced information
        bus_stop_usage = assignments_df['assigned_stop_id'].value_counts()
        
        for _, stop in self.bus_stops_data.iterrows():
            usage_count = bus_stop_usage.get(stop['cluster_id'], 0)
            
            folium.CircleMarker(
                location=[float(stop['latitude']), float(stop['longitude'])],
                radius=max(8, min(20, usage_count * 2)),
                popup=folium.Popup(f"""
                    <b>Bus Stop {stop['cluster_id']}</b><br>
                    Capacity: {stop['student_count']} students<br>
                    Assigned: {usage_count} students<br>
                    Utilization: {(usage_count/stop['student_count']*100):.1f}%
                """, max_width=200),
                color='navy',
                fill=True,
                fillColor='lightblue' if usage_count > 0 else 'gray',
                fillOpacity=0.8,
                weight=2
            ).add_to(m)
        
        # Add student assignments with cluster colors
        cluster_ids = assignments_df['cluster_id'].unique()
        cluster_color_map = {cid: colors[i % len(colors)] for i, cid in enumerate(cluster_ids)}
        
        for _, assignment in assignments_df.iterrows():
            cluster_color = cluster_color_map.get(assignment['cluster_id'], 'gray')
            
            # Student marker
            folium.CircleMarker(
                location=[assignment['student_lat'], assignment['student_lon']],
                radius=4,
                popup=folium.Popup(f"""
                    <b>Student {assignment['user']}</b><br>
                    Cluster: {assignment['cluster_id']}<br>
                    Assigned Stop: {assignment['assigned_stop_id']}<br>
                    Distance: {assignment['road_distance_km']:.3f} km<br>
                    Within Limit: {'Yes' if assignment['within_distance_limit'] else 'No'}
                """, max_width=200),
                color=cluster_color,
                fill=True,
                fillColor=cluster_color,
                fillOpacity=0.7
            ).add_to(m)
            
            # Connection line
            line_color = 'green' if assignment['within_distance_limit'] else 'red'
            folium.PolyLine(
                locations=[
                    [assignment['student_lat'], assignment['student_lon']],
                    [assignment['assigned_stop_lat'], assignment['assigned_stop_lon']]
                ],
                color=line_color,
                weight=1,
                opacity=0.6
            ).add_to(m)
        
        # Add legend with API efficiency info
        api_efficiency = f"API Calls: {self.api_calls_made} | Cache Hits: {self.cache_hits}"
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 280px; min-height: 140px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px">
        <p><b>Ultra-Optimized Assignment</b></p>
        <p style="font-size:10px; color: blue;"><b>{api_efficiency}</b></p>
        <p><i class="fa fa-circle" style="color:lightblue"></i> Used Bus Stops</p>
        <p><i class="fa fa-circle" style="color:gray"></i> Unused Bus Stops</p>
        <p><i class="fa fa-circle" style="color:red"></i> Students (by cluster)</p>
        <p><i class="fa fa-minus" style="color:green"></i> Valid (≤{self.max_distance_km}km)</p>
        <p><i class="fa fa-minus" style="color:red"></i> Invalid (>{self.max_distance_km}km)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        m.save(output_path)
        print(f"Ultra-optimized visualization saved to {output_path}")
    
    def generate_comprehensive_report(self, assignments_df, cluster_summary_df, 
                                    output_prefix="ultra_optimized_boarding_assignment"):
        """
        Generate comprehensive assignment reports with API efficiency metrics
        """
        if len(assignments_df) == 0:
            print("No assignments to report")
            return
        
        # Save detailed assignments
        assignments_df.to_csv(f"{output_prefix}_detailed.csv", index=False)
        
        # Save cluster summary
        cluster_summary_df.to_csv(f"{output_prefix}_cluster_summary.csv", index=False)
        
        # Generate statistics
        total_students = len(self.student_data)
        assigned_students = len(assignments_df)
        within_limit = len(assignments_df[assignments_df['within_distance_limit']])
        
        avg_distance = assignments_df['road_distance_km'].mean()
        median_distance = assignments_df['road_distance_km'].median()
        max_distance = assignments_df['road_distance_km'].max()
        min_distance = assignments_df['road_distance_km'].min()
        
        unique_stops_used = assignments_df['assigned_stop_id'].nunique()
        total_stops_available = len(self.bus_stops_data)
        
        # Cluster statistics
        avg_cluster_size = cluster_summary_df['student_count'].mean() if len(cluster_summary_df) > 0 else 0
        avg_reachability = cluster_summary_df['reachability_percentage'].mean() if len(cluster_summary_df) > 0 else 0
        
        # API efficiency metrics
        total_possible_pairs = len(self.student_data) * len(self.bus_stops_data)
        api_efficiency = (1 - self.api_calls_made / max(1, total_possible_pairs)) * 100
        cache_efficiency = self.cache_hits / max(1, self.cache_hits + self.api_calls_made) * 100
        
        print("\n" + "="*70)
        print("ULTRA-OPTIMIZED BOARDING POINT ASSIGNMENT REPORT")
        print("="*70)
        print(f"Algorithm: HDBSCAN + Smart Pre-filtering + Persistent Caching")
        print(f"Distance Constraint: {self.max_distance_km} km (road distance)")
        print()
        
        print("🚀 API EFFICIENCY METRICS:")
        print(f"  Total possible student-stop pairs: {total_possible_pairs:,}")
        print(f"  API calls made: {self.api_calls_made:,}")
        print(f"  Cache hits: {self.cache_hits:,}")
        print(f"  API call reduction: {api_efficiency:.2f}%")
        print(f"  Cache hit rate: {cache_efficiency:.1f}%")
        print(f"  Estimated cost savings: ${(total_possible_pairs - self.api_calls_made) * 0.005:.2f}")
        print()
        
        print("📊 ASSIGNMENT SUMMARY:")
        print(f"  Total students in dataset: {total_students}")
        print(f"  Students assigned: {assigned_students} ({assigned_students/total_students*100:.1f}%)")
        print(f"  Students within distance limit: {within_limit} ({within_limit/assigned_students*100:.1f}%)")
        print(f"  Students beyond distance limit: {assigned_students - within_limit}")
        print()
        
        print("📏 DISTANCE STATISTICS:")
        print(f"  Average distance: {avg_distance:.3f} km")
        print(f"  Median distance: {median_distance:.3f} km")
        print(f"  Minimum distance: {min_distance:.3f} km")
        print(f"  Maximum distance: {max_distance:.3f} km")
        print()
        
        print("🚌 BUS STOP UTILIZATION:")
        print(f"  Total bus stops available: {total_stops_available}")
        print(f"  Bus stops used: {unique_stops_used} ({unique_stops_used/total_stops_available*100:.1f}%)")
        print()
        
        if len(cluster_summary_df) > 0:
            print("🎯 CLUSTERING STATISTICS:")
            print(f"  Number of clusters: {len(cluster_summary_df)}")
            print(f"  Average cluster size: {avg_cluster_size:.1f} students")
            print(f"  Average reachability: {avg_reachability:.1f}%")
            print()
        
        # Distance distribution
        print("📈 DISTANCE DISTRIBUTION:")
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, float('inf')]
        for i in range(len(bins)-1):
            if bins[i+1] == float('inf'):
                count = len(assignments_df[assignments_df['road_distance_km'] > bins[i]])
                label = f"  > {bins[i]:.1f} km"
            else:
                count = len(assignments_df[
                    (assignments_df['road_distance_km'] > bins[i]) & 
                    (assignments_df['road_distance_km'] <= bins[i+1])
                ])
                label = f"  {bins[i]:.1f}-{bins[i+1]:.1f} km"
            
            percentage = count / assigned_students * 100 if assigned_students > 0 else 0
            print(f"{label}: {count} students ({percentage:.1f}%)")
        
        print()
        print("🏆 TOP 5 MOST UTILIZED BUS STOPS:")
        stop_usage = assignments_df['assigned_stop_id'].value_counts().head()
        for stop_id, count in stop_usage.items():
            stop_info = self.bus_stops_data[self.bus_stops_data['cluster_id'] == stop_id].iloc[0]
            utilization = count / stop_info['student_count'] * 100
            print(f"  Stop {stop_id}: {count} students assigned (Capacity: {stop_info['student_count']}, {utilization:.1f}% utilized)")
        
        print()
        print("💾 CACHE STATISTICS:")
        print(f"  Total cached distances: {len(self.distance_cache):,}")
        print(f"  Cache file: {self.cache_file}")
        print(f"  Estimated cache size: {len(str(self.distance_cache)) / 1024:.1f} KB")
        
        print()
        print(f"📁 FILES GENERATED:")
        print(f"  Detailed assignments: {output_prefix}_detailed.csv")
        print(f"  Cluster summary: {output_prefix}_cluster_summary.csv")
        print(f"  Distance cache: {self.cache_file}")
        print("="*70)

def main():
    import dotenv
    dotenv.load_dotenv()
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    """
    Main function to run the ultra-optimized boarding point assignment
    """
    # Configuration
    MAX_DISTANCE_KM = 1.0
    MIN_CLUSTER_SIZE = 8
    MIN_SAMPLES = 5
    CACHE_FILE = "ultra_optimized_distance_cache.pkl"
    
    try:
        # Initialize the ultra-optimized assignment system
        print("🚀 Initializing Ultra-Optimized Boarding Point Assignment System...")
        assignment_system = OptimizedBoardingPointAssignment(
            student_data_path="all workinig/Output Data/cleaned_student_locations_20250807_110931.csv",
            bus_stops_path="all workinig/Output Data/bus_stops_20250731_013718.csv",
            google_maps_api_key=GOOGLE_MAPS_API_KEY,
            max_distance_km=MAX_DISTANCE_KM,
            cache_file=CACHE_FILE
        )
        
        print(f"\n⚡ Starting ultra-optimized assignment with minimal API calls...")
        print(f"   Target: <{len(assignment_system.student_data) * len(assignment_system.bus_stops_data) * 0.01:.0f} API calls")
        print(f"   (vs {len(assignment_system.student_data) * len(assignment_system.bus_stops_data):,} naive approach)")
        
        # Run the ultra-optimized assignment
        assignments_df, cluster_summary_df = assignment_system.assign_boarding_points_optimized(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES
        )
        
        if len(assignments_df) > 0:
            # Generate comprehensive report with API efficiency metrics
            assignment_system.generate_comprehensive_report(
                assignments_df, cluster_summary_df
            )
            
            # Create enhanced visualization
            assignment_system.create_enhanced_visualization(
                assignments_df, cluster_summary_df
            )
            
            # Generate optimization recommendations
            generate_ultra_optimization_recommendations(assignments_df, cluster_summary_df, assignment_system)
            
            print(f"\n✅ SUCCESS! Processed {len(assignments_df)} assignments with {assignment_system.api_calls_made} API calls!")
            print(f"💰 Estimated savings: ${(len(assignment_system.student_data) * len(assignment_system.bus_stops_data) - assignment_system.api_calls_made) * 0.005:.2f}")
            
        else:
            print("❌ No assignments were made. Please check:")
            print("- Distance constraints are not too restrictive")
            print("- Student and bus stop data are valid")
            print("- Google Maps API key is working")
            
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("Please check your configuration and data files.")

def generate_ultra_optimization_recommendations(assignments_df, cluster_summary_df, assignment_system):
    """
    Generate optimization recommendations with API efficiency focus
    """
    print(f"\n🎯 ULTRA-OPTIMIZATION RECOMMENDATIONS:")
    print("-" * 50)
    
    # API Efficiency recommendations
    api_call_ratio = assignment_system.api_calls_made / max(1, len(assignment_system.student_data) * len(assignment_system.bus_stops_data))
    cache_hit_rate = assignment_system.cache_hits / max(1, assignment_system.cache_hits + assignment_system.api_calls_made)
    
    print("🚀 API EFFICIENCY:")
    if api_call_ratio < 0.05:
        print(f"   ✅ Excellent API efficiency: {api_call_ratio*100:.3f}% of possible calls made")
    elif api_call_ratio < 0.20:
        print(f"   ⚠️  Good API efficiency: {api_call_ratio*100:.1f}% of possible calls made")
    else:
        print(f"   ❌ Consider further optimization: {api_call_ratio*100:.1f}% of possible calls made")
        print("   💡 Try increasing pre-filtering aggressiveness or cluster size")
    
    if cache_hit_rate > 0.5:
        print(f"   ✅ Good cache utilization: {cache_hit_rate*100:.1f}% hit rate")
    else:
        print(f"   💡 Low cache hit rate: {cache_hit_rate*100:.1f}% - consider running multiple times to build cache")
    
    # Standard optimization recommendations (condensed)
    print(f"\n📊 ASSIGNMENT QUALITY:")
    
    # Coverage analysis
    total_students = len(assignment_system.student_data)
    assigned_students = len(assignments_df)
    coverage_rate = assigned_students / total_students * 100
    
    if coverage_rate >= 95:
        print(f"   ✅ Excellent coverage: {coverage_rate:.1f}% of students assigned")
    else:
        print(f"   ⚠️  Coverage could be improved: {coverage_rate:.1f}% of students assigned")
        print(f"   💡 {total_students - assigned_students} students unassigned - consider increasing distance limit")
    
    # Distance quality
    within_limit = len(assignments_df[assignments_df['within_distance_limit']])
    distance_compliance = within_limit / len(assignments_df) * 100 if len(assignments_df) > 0 else 0
    
    if distance_compliance >= 90:
        print(f"   ✅ Good distance compliance: {distance_compliance:.1f}% within limit")
    else:
        print(f"   ⚠️  Distance compliance: {distance_compliance:.1f}% within {assignment_system.max_distance_km}km limit")
    
    # Utilization analysis
    stop_usage = assignments_df['assigned_stop_id'].value_counts()
    overutilized = sum(1 for stop_id, usage in stop_usage.items() 
                      if usage > assignment_system.bus_stops_data[
                          assignment_system.bus_stops_data['cluster_id'] == stop_id
                      ].iloc[0]['student_count'] * 0.9)
    
    if overutilized > 0:
        print(f"   ⚠️  {overutilized} bus stops are >90% utilized")
        print("   💡 Consider capacity increases or additional stops")
    
    unused_stops = len(assignment_system.bus_stops_data) - assignments_df['assigned_stop_id'].nunique()
    if unused_stops > len(assignment_system.bus_stops_data) * 0.3:
        print(f"   💡 {unused_stops} bus stops unused - consider reallocation")
    
    print(f"\n⚡ PERFORMANCE SUMMARY:")
    print(f"   API calls made: {assignment_system.api_calls_made:,}")
    print(f"   Cache entries: {len(assignment_system.distance_cache):,}")
    print(f"   Processing efficiency: {len(assignments_df)/max(1,assignment_system.api_calls_made):.1f} assignments/API call")
    
    print(f"\n🔄 FOR NEXT RUN:")
    print("   - Cache will provide even better performance")
    print("   - Consider adjusting cluster parameters if needed")
    print("   - Monitor API quotas and costs")
    print(f"   - Current cache file: {assignment_system.cache_file}")

def validate_api_key_efficient():
    """
    Validate Google Maps API key with minimal test call
    """
    import googlemaps
    
    api_key = input("Please enter your Google Maps API key: ").strip()
    
    if not api_key or api_key == "YOUR_GOOGLE_MAPS_API_KEY_HERE":
        print("❌ Invalid API key. Please provide a valid Google Maps API key.")
        print("💡 Get your API key from: https://developers.google.com/maps/documentation/distance-matrix/get-api-key")
        return None
    
    try:
        gmaps = googlemaps.Client(key=api_key)
        
        # Minimal test with just one pair
        test_result = gmaps.distance_matrix(
            origins=[(40.7128, -74.0060)],  # New York
            destinations=[(40.7589, -73.9851)],  # Times Square
            mode="driving",
            units="metric"
        )
        
        if test_result['status'] == 'OK':
            print("✅ Google Maps API key is valid and working!")
            print("⚡ Ultra-optimization mode: Minimal API calls ahead!")
            return api_key
        else:
            print("❌ API key validation failed. Please check your key and billing settings.")
            return None
            
    except Exception as e:
        print(f"❌ API validation error: {e}")
        print("Please check your API key and internet connection.")
        return None

def main_with_validation():
    """
    Main function with interactive API key validation and ultra-optimization focus
    """
    print("🚀 ULTRA-OPTIMIZED Boarding Point Assignment System")
    print("=" * 60)
    print("🎯 Focus: MINIMAL API CALLS | MAXIMUM EFFICIENCY")
    print("=" * 60)
    
    # Validate API key first
    api_key = validate_api_key_efficient()
    if not api_key:
        return
    
    # Configuration
    MAX_DISTANCE_KM = 1.0
    MIN_CLUSTER_SIZE = 8
    MIN_SAMPLES = 5
    CACHE_FILE = "ultra_optimized_distance_cache.pkl"
    
    try:
        print(f"\n⚡ Initializing ultra-efficient system...")
        assignment_system = OptimizedBoardingPointAssignment(
            student_data_path="cleaned_student_locations_20250731_010007.csv",
            bus_stops_path="bus_stops_20250731_013718.csv",
            google_maps_api_key=api_key,
            max_distance_km=MAX_DISTANCE_KM,
            cache_file=CACHE_FILE
        )
        
        # Estimate API calls before starting
        total_possible = len(assignment_system.student_data) * len(assignment_system.bus_stops_data)
        estimated_calls = total_possible * 0.01  # Targeting <1% of possible calls
        
        print(f"📊 Efficiency Target:")
        print(f"   Naive approach would need: {total_possible:,} API calls")
        print(f"   Our target: <{estimated_calls:.0f} API calls ({estimated_calls/total_possible*100:.2f}%)")
        print(f"   Estimated savings: ${(total_possible - estimated_calls) * 0.005:.2f}")
        
        print(f"\n🚀 Running ultra-optimized assignment...")
        assignments_df, cluster_summary_df = assignment_system.assign_boarding_points_optimized(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES
        )
        
        if len(assignments_df) > 0:
            # Generate all outputs with efficiency metrics
            assignment_system.generate_comprehensive_report(assignments_df, cluster_summary_df)
            assignment_system.create_enhanced_visualization(assignments_df, cluster_summary_df)
            generate_ultra_optimization_recommendations(assignments_df, cluster_summary_df, assignment_system)
            
            actual_efficiency = assignment_system.api_calls_made / total_possible * 100
            print(f"\n🎉 MISSION ACCOMPLISHED!")
            print(f"   Actual API calls: {assignment_system.api_calls_made:,}")
            print(f"   Efficiency achieved: {actual_efficiency:.3f}% of possible calls")
            print(f"   Actual savings: ${(total_possible - assignment_system.api_calls_made) * 0.005:.2f}")
            
            if actual_efficiency < 1.0:
                print("   🏆 ULTRA-EFFICIENT: <1% of possible API calls!")
            elif actual_efficiency < 5.0:
                print("   ✅ HIGHLY EFFICIENT: <5% of possible API calls!")
            else:
                print("   ⚠️  Room for improvement in efficiency")
            
        else:
            print("❌ No assignments made. Check your constraints and data.")
            
    except Exception as e:
        print(f"❌ Execution error: {e}")

# Additional utility functions for cache management
def analyze_cache(cache_file="ultra_optimized_distance_cache.pkl"):
    """
    Analyze the distance cache for insights
    """
    if not os.path.exists(cache_file):
        print(f"Cache file {cache_file} not found")
        return
    
    try:
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        
        print(f"📊 CACHE ANALYSIS:")
        print(f"   Total entries: {len(cache):,}")
        print(f"   Estimated size: {len(str(cache)) / 1024:.1f} KB")
        
        # Analyze distances
        distances = [v for v in cache.values() if v is not None]
        if distances:
            print(f"   Distance range: {min(distances):.3f} - {max(distances):.3f} km")
            print(f"   Average distance: {sum(distances)/len(distances):.3f} km")
        
        print(f"   Cache file: {cache_file}")
        
    except Exception as e:
        print(f"Error analyzing cache: {e}")

def clear_cache(cache_file="ultra_optimized_distance_cache.pkl"):
    """
    Clear the distance cache (use with caution!)
    """
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"✅ Cache {cache_file} cleared")
    else:
        print(f"Cache file {cache_file} not found")

if __name__ == "__main__":
    # Choose which main function to run
    main()  # Use this if you want to hardcode the API key
    # main_with_validation()  # Use this for interactive API key validation
    
    # Utility functions (uncomment to use):
    # analyze_cache()  # Analyze cache contents
    # clear_cache()    # Clear cache if needed