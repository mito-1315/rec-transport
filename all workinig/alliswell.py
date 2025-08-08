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

class OptimizedBoardingPointAssignment:
    def __init__(self, student_data_path, bus_stops_path, google_maps_api_key, max_distance_km=1.0):
        """
        Initialize the optimized boarding point assignment system with HDBSCAN and Google Maps API
        
        Args:
            student_data_path: Path to student locations CSV
            bus_stops_path: Path to bus stops CSV
            google_maps_api_key: Google Maps API key
            max_distance_km: Maximum road distance in km (default 1.0km)
        """
        self.max_distance_km = max_distance_km
        self.student_data = pd.read_csv(student_data_path)
        self.bus_stops_data = pd.read_csv(bus_stops_path)
        
        # Initialize Google Maps client
        self.gmaps = googlemaps.Client(key=google_maps_api_key)
        
        # Cache for distance calculations to avoid repeated API calls
        self.distance_cache = {}
        
        print(f"Loaded {len(self.student_data)} student locations")
        print(f"Loaded {len(self.bus_stops_data)} bus stops")
        print(f"Maximum distance constraint: {self.max_distance_km} km")
        
    def calculate_road_distance_batch(self, origins, destinations, max_batch_size=25):
        """
        Calculate road distances using Google Maps Distance Matrix API in batches
        
        Args:
            origins: List of (lat, lon) tuples for origins
            destinations: List of (lat, lon) tuples for destinations
            max_batch_size: Maximum batch size for API calls
            
        Returns:
            Dictionary with distances in km
        """
        distances = {}
        
        # Process in batches to respect API limits
        for i in range(0, len(origins), max_batch_size):
            origin_batch = origins[i:i + max_batch_size]
            
            for j in range(0, len(destinations), max_batch_size):
                dest_batch = destinations[j:j + max_batch_size]
                
                try:
                    # Create cache keys
                    cache_keys = []
                    uncached_origins = []
                    uncached_destinations = []
                    
                    for orig_idx, origin in enumerate(origin_batch):
                        for dest_idx, dest in enumerate(dest_batch):
                            cache_key = f"{origin[0]:.6f},{origin[1]:.6f}|{dest[0]:.6f},{dest[1]:.6f}"
                            cache_keys.append((i + orig_idx, j + dest_idx, cache_key))
                            
                            if cache_key not in self.distance_cache:
                                if origin not in uncached_origins:
                                    uncached_origins.append(origin)
                                if dest not in uncached_destinations:
                                    uncached_destinations.append(dest)
                    
                    # Only make API call if we have uncached distances
                    if uncached_origins and uncached_destinations:
                        matrix = self.gmaps.distance_matrix(
                            origins=uncached_origins,
                            destinations=uncached_destinations,
                            mode="driving",
                            units="metric",
                            avoid="tolls"
                        )
                        
                        # Cache the results
                        if matrix['status'] == 'OK':
                            for orig_idx, origin in enumerate(uncached_origins):
                                for dest_idx, destination in enumerate(uncached_destinations):
                                    element = matrix['rows'][orig_idx]['elements'][dest_idx]
                                    cache_key = f"{origin[0]:.6f},{origin[1]:.6f}|{destination[0]:.6f},{destination[1]:.6f}"
                                    
                                    if element['status'] == 'OK':
                                        distance_km = element['distance']['value'] / 1000.0
                                        self.distance_cache[cache_key] = distance_km
                                    else:
                                        self.distance_cache[cache_key] = None
                    
                    # Retrieve distances from cache
                    for orig_idx, dest_idx, cache_key in cache_keys:
                        if cache_key in self.distance_cache:
                            distances[(orig_idx, dest_idx)] = self.distance_cache[cache_key]
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error in distance calculation batch: {e}")
                    continue
        
        return distances
    
    def calculate_straight_line_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate straight-line distance using geodesic distance
        """
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    
    def create_student_clusters_hdbscan(self, min_cluster_size=8, min_samples=5):
        """
        Create student clusters using HDBSCAN algorithm
        
        Args:
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for core points
            
        Returns:
            Cluster labels and cluster information
        """
        print("Creating student clusters using HDBSCAN...")
        
        # Prepare coordinate data
        student_coords = self.student_data[['latitude', 'longitude']].astype(float).values
        
        # Convert to radians for better clustering (since we're dealing with geographic coordinates)
        student_coords_rad = np.radians(student_coords)
        
        # Apply HDBSCAN clustering
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='haversine'  # Great for geographic coordinates
        )
        
        cluster_labels = clusterer.fit_predict(student_coords_rad)
        
        # Analyze clusters
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        print(f"Silhouette-like score: {clusterer.relative_validity_:.3f}" if hasattr(clusterer, 'relative_validity_') else "")
        
        # Add cluster labels to student data
        self.student_data['cluster_id'] = cluster_labels
        
        return cluster_labels, clusterer
    
    def find_optimal_centroid_for_cluster(self, cluster_students):
        """
        Find the optimal centroid bus stop for a cluster of students
        
        Args:
            cluster_students: DataFrame of students in the cluster
            
        Returns:
            Optimal bus stop information
        """
        if len(cluster_students) == 0:
            return None
        
        # Calculate cluster geographic centroid
        centroid_lat = cluster_students['latitude'].astype(float).mean()
        centroid_lon = cluster_students['longitude'].astype(float).mean()
        
        # Get student coordinates
        student_coords = [(float(row['latitude']), float(row['longitude'])) 
                         for _, row in cluster_students.iterrows()]
        
        # Get bus stop coordinates
        bus_stop_coords = [(float(row['latitude']), float(row['longitude'])) 
                          for _, row in self.bus_stops_data.iterrows()]
        
        # Pre-filter bus stops by straight-line distance (2x the road distance limit)
        candidate_stops = []
        for idx, (stop_lat, stop_lon) in enumerate(bus_stop_coords):
            straight_distance = self.calculate_straight_line_distance(
                centroid_lat, centroid_lon, stop_lat, stop_lon
            )
            if straight_distance <= self.max_distance_km * 2:  # Pre-filter
                candidate_stops.append({
                    'index': idx,
                    'coordinates': (stop_lat, stop_lon),
                    'straight_distance': straight_distance,
                    'stop_data': self.bus_stops_data.iloc[idx]
                })
        
        if not candidate_stops:
            print(f"No candidate bus stops found within {self.max_distance_km * 2} km straight-line distance")
            return None
        
        # Sort by straight-line distance and take top candidates
        candidate_stops.sort(key=lambda x: x['straight_distance'])
        top_candidates = candidate_stops[:min(10, len(candidate_stops))]
        
        print(f"Evaluating {len(top_candidates)} candidate bus stops for cluster...")
        
        # Calculate road distances from all students to top candidate stops
        candidate_coords = [stop['coordinates'] for stop in top_candidates]
        
        # Batch calculate distances
        distances = self.calculate_road_distance_batch(student_coords, candidate_coords)
        
        # Evaluate each candidate stop
        best_stop = None
        best_score = -1
        
        for stop_idx, candidate in enumerate(top_candidates):
            reachable_students = 0
            total_distance = 0
            max_student_distance = 0
            
            # Check how many students can reach this stop
            for student_idx in range(len(student_coords)):
                distance_key = (student_idx, stop_idx)
                road_distance = distances.get(distance_key)
                
                if road_distance is None:
                    # Fallback to straight-line distance
                    student_coord = student_coords[student_idx]
                    road_distance = self.calculate_straight_line_distance(
                        student_coord[0], student_coord[1],
                        candidate['coordinates'][0], candidate['coordinates'][1]
                    )
                
                if road_distance <= self.max_distance_km:
                    reachable_students += 1
                    total_distance += road_distance
                    max_student_distance = max(max_student_distance, road_distance)
            
            # Calculate score based on:
            # 1. Percentage of students that can reach the stop (most important)
            # 2. Average distance (lower is better)
            # 3. Maximum distance (lower is better)
            # 4. Bus stop capacity
            
            if reachable_students > 0:
                reachability_score = reachable_students / len(student_coords)
                avg_distance = total_distance / reachable_students
                capacity_score = min(candidate['stop_data']['student_count'] / 50.0, 1.0)
                
                # Weighted scoring
                score = (
                    reachability_score * 100 +  # 100 points for full reachability
                    (1 - avg_distance / self.max_distance_km) * 20 +  # 20 points for short average distance
                    (1 - max_student_distance / self.max_distance_km) * 10 +  # 10 points for short max distance
                    capacity_score * 5  # 5 points for capacity
                )
                
                if score > best_score:
                    best_score = score
                    best_stop = {
                        'stop_id': candidate['stop_data']['cluster_id'],
                        'stop_lat': candidate['coordinates'][0],
                        'stop_lon': candidate['coordinates'][1],
                        'student_count': candidate['stop_data']['student_count'],
                        'reachable_students': reachable_students,
                        'total_students': len(student_coords),
                        'avg_distance': avg_distance,
                        'max_distance': max_student_distance,
                        'reachability_percentage': reachability_score * 100,
                        'score': score
                    }
        
        return best_stop
    
    def assign_boarding_points_optimized(self, min_cluster_size=8, min_samples=5):
        """
        Optimized boarding point assignment using HDBSCAN clustering
        
        Args:
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for core points
            
        Returns:
            DataFrame with assignment results
        """
        print("Starting optimized boarding point assignment...")
        
        # Create clusters
        cluster_labels, clusterer = self.create_student_clusters_hdbscan(
            min_cluster_size, min_samples
        )
        
        assignments = []
        cluster_summary = []
        
        # Process each cluster
        unique_clusters = set(cluster_labels)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)  # Remove noise cluster
        
        print(f"Processing {len(unique_clusters)} clusters...")
        
        for cluster_id in tqdm(unique_clusters, desc="Processing clusters"):
            cluster_mask = cluster_labels == cluster_id
            cluster_students = self.student_data[cluster_mask].copy()
            
            print(f"\nCluster {cluster_id}: {len(cluster_students)} students")
            
            # Find optimal boarding point for this cluster
            optimal_stop = self.find_optimal_centroid_for_cluster(cluster_students)
            
            if optimal_stop:
                print(f"  Assigned to stop {optimal_stop['stop_id']}")
                print(f"  Reachability: {optimal_stop['reachable_students']}/{optimal_stop['total_students']} students ({optimal_stop['reachability_percentage']:.1f}%)")
                print(f"  Avg distance: {optimal_stop['avg_distance']:.3f} km")
                print(f"  Max distance: {optimal_stop['max_distance']:.3f} km")
                
                # Calculate actual distances for each student in cluster
                student_coords = [(float(row['latitude']), float(row['longitude'])) 
                                for _, row in cluster_students.iterrows()]
                stop_coords = [(optimal_stop['stop_lat'], optimal_stop['stop_lon'])]
                
                # Get road distances
                distances = self.calculate_road_distance_batch(student_coords, stop_coords)
                
                # Assign boarding point to students
                for idx, (_, student) in enumerate(cluster_students.iterrows()):
                    distance_key = (idx, 0)
                    road_distance = distances.get(distance_key)
                    
                    if road_distance is None:
                        # Fallback to straight-line distance
                        road_distance = self.calculate_straight_line_distance(
                            float(student['latitude']), float(student['longitude']),
                            optimal_stop['stop_lat'], optimal_stop['stop_lon']
                        )
                    
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
            else:
                print(f"  No suitable boarding point found")
        
        # Handle noise points (students not in any cluster)
        noise_mask = cluster_labels == -1
        noise_students = self.student_data[noise_mask]
        
        if len(noise_students) > 0:
            print(f"\nProcessing {len(noise_students)} noise points individually...")
            
            for _, student in noise_students.iterrows():
                # Find closest bus stop for individual students
                student_coord = [(float(student['latitude']), float(student['longitude']))]
                bus_stop_coords = [(float(row['latitude']), float(row['longitude'])) 
                                  for _, row in self.bus_stops_data.iterrows()]
                
                # Pre-filter by straight-line distance
                close_stops = []
                for idx, stop_coord in enumerate(bus_stop_coords):
                    straight_distance = self.calculate_straight_line_distance(
                        student_coord[0][0], student_coord[0][1],
                        stop_coord[0], stop_coord[1]
                    )
                    if straight_distance <= self.max_distance_km * 1.5:
                        close_stops.append((idx, stop_coord))
                
                if close_stops:
                    close_stop_coords = [coord for _, coord in close_stops]
                    distances = self.calculate_road_distance_batch(student_coord, close_stop_coords)
                    
                    # Find closest accessible stop
                    best_distance = float('inf')
                    best_stop_idx = None
                    
                    for i, (original_idx, _) in enumerate(close_stops):
                        distance = distances.get((0, i))
                        if distance and distance <= self.max_distance_km and distance < best_distance:
                            best_distance = distance
                            best_stop_idx = original_idx
                    
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
                            'cluster_id': -1,  # Noise cluster
                            'within_distance_limit': True
                        })
        
        assignments_df = pd.DataFrame(assignments)
        cluster_summary_df = pd.DataFrame(cluster_summary)
        
        return assignments_df, cluster_summary_df
    
    def create_enhanced_visualization(self, assignments_df, cluster_summary_df, output_path="optimized_boarding_assignments_map.html"):
        """
        Create an enhanced interactive map with cluster visualization
        """
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
                radius=max(8, min(20, usage_count * 2)),  # Size based on usage
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
        
        # Add cluster centroids
        for _, cluster_info in cluster_summary_df.iterrows():
            cluster_students = assignments_df[assignments_df['cluster_id'] == cluster_info['cluster_id']]
            if len(cluster_students) > 0:
                centroid_lat = cluster_students['student_lat'].mean()
                centroid_lon = cluster_students['student_lon'].mean()
                
                folium.Marker(
                    location=[centroid_lat, centroid_lon],
                    popup=folium.Popup(f"""
                        <b>Cluster {cluster_info['cluster_id']} Centroid</b><br>
                        Students: {cluster_info['student_count']}<br>
                        Reachable: {cluster_info['reachable_students']} ({cluster_info['reachability_percentage']:.1f}%)<br>
                        Avg Distance: {cluster_info['avg_distance']:.3f} km<br>
                        Max Distance: {cluster_info['max_distance']:.3f} km
                    """, max_width=250),
                    icon=folium.Icon(color='darkblue', icon='info-sign')
                ).add_to(m)
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 250px; min-height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px">
        <p><b>Optimized Boarding Point Assignment</b></p>
        <p><i class="fa fa-circle" style="color:lightblue"></i> Used Bus Stops</p>
        <p><i class="fa fa-circle" style="color:gray"></i> Unused Bus Stops</p>
        <p><i class="fa fa-circle" style="color:red"></i> Students (colored by cluster)</p>
        <p><i class="fa fa-minus" style="color:green"></i> Valid Assignments (≤{self.max_distance_km}km)</p>
        <p><i class="fa fa-minus" style="color:red"></i> Invalid Assignments (>{self.max_distance_km}km)</p>
        <p><i class="fa fa-map-marker" style="color:darkblue"></i> Cluster Centroids</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        m.save(output_path)
        print(f"Enhanced visualization saved to {output_path}")
    
    def generate_comprehensive_report(self, assignments_df, cluster_summary_df, 
                                    output_prefix="optimized_boarding_assignment"):
        """
        Generate comprehensive assignment reports and analysis
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
        
        print("\n" + "="*60)
        print("OPTIMIZED BOARDING POINT ASSIGNMENT REPORT")
        print("="*60)
        print(f"Algorithm: HDBSCAN Clustering + Google Maps API")
        print(f"Distance Constraint: {self.max_distance_km} km (road distance)")
        print()
        
        print("ASSIGNMENT SUMMARY:")
        print(f"  Total students in dataset: {total_students}")
        print(f"  Students assigned: {assigned_students} ({assigned_students/total_students*100:.1f}%)")
        print(f"  Students within distance limit: {within_limit} ({within_limit/assigned_students*100:.1f}%)")
        print(f"  Students beyond distance limit: {assigned_students - within_limit}")
        print()
        
        print("DISTANCE STATISTICS:")
        print(f"  Average distance: {avg_distance:.3f} km")
        print(f"  Median distance: {median_distance:.3f} km")
        print(f"  Minimum distance: {min_distance:.3f} km")
        print(f"  Maximum distance: {max_distance:.3f} km")
        print()
        
        print("BUS STOP UTILIZATION:")
        print(f"  Total bus stops available: {total_stops_available}")
        print(f"  Bus stops used: {unique_stops_used} ({unique_stops_used/total_stops_available*100:.1f}%)")
        print()
        
        if len(cluster_summary_df) > 0:
            print("CLUSTERING STATISTICS:")
            print(f"  Number of clusters: {len(cluster_summary_df)}")
            print(f"  Average cluster size: {avg_cluster_size:.1f} students")
            print(f"  Average reachability: {avg_reachability:.1f}%")
            print()
        
        # Distance distribution
        print("DISTANCE DISTRIBUTION:")
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
        print("TOP 5 MOST UTILIZED BUS STOPS:")
        stop_usage = assignments_df['assigned_stop_id'].value_counts().head()
        for stop_id, count in stop_usage.items():
            stop_info = self.bus_stops_data[self.bus_stops_data['cluster_id'] == stop_id].iloc[0]
            utilization = count / stop_info['student_count'] * 100
            print(f"  Stop {stop_id}: {count} students assigned (Capacity: {stop_info['student_count']}, {utilization:.1f}% utilized)")
        
        print()
        print(f"Detailed report saved to: {output_prefix}_detailed.csv")
        print(f"Cluster summary saved to: {output_prefix}_cluster_summary.csv")
        print("="*60)

def main():
    """
    Main function to run the optimized boarding point assignment
    """
    # Configuration
    GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY_HERE"  # Replace with your actual API key
    MAX_DISTANCE_KM = 1.0
    MIN_CLUSTER_SIZE = 8
    MIN_SAMPLES = 5
    
    try:
        # Initialize the optimized assignment system
        print("Initializing Optimized Boarding Point Assignment System...")
        assignment_system = OptimizedBoardingPointAssignment(
            student_data_path="cleaned_student_locations_20250731_010007.csv",
            bus_stops_path="bus_stops_20250731_013718.csv",
            google_maps_api_key=GOOGLE_MAPS_API_KEY,
            max_distance_km=MAX_DISTANCE_KM
        )
        
        # Run the optimized assignment
        print("\nStarting optimized boarding point assignment with HDBSCAN clustering...")
        assignments_df, cluster_summary_df = assignment_system.assign_boarding_points_optimized(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES
        )
        
        if len(assignments_df) > 0:
            # Generate comprehensive report
            assignment_system.generate_comprehensive_report(
                assignments_df, cluster_summary_df
            )
            
            # Create enhanced visualization
            assignment_system.create_enhanced_visualization(
                assignments_df, cluster_summary_df
            )
            
            # Additional analysis and recommendations
            print("\nGENERATING RECOMMENDATIONS...")
            generate_optimization_recommendations(assignments_df, cluster_summary_df, assignment_system)
            
            print(f"\n✅ Successfully processed {len(assignments_df)} student assignments!")
            print("📊 Check the generated files for detailed results and visualizations.")
            
        else:
            print("❌ No assignments were made. Please check:")
            print("- Distance constraints are not too restrictive")
            print("- Student and bus stop data are valid")
            print("- Google Maps API key is working")
            
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("Please check your configuration and data files.")

def generate_optimization_recommendations(assignments_df, cluster_summary_df, assignment_system):
    """
    Generate optimization recommendations based on assignment results
    """
    print("\nOPTIMIZATION RECOMMENDATIONS:")
    print("-" * 40)
    
    # 1. Analyze poorly served clusters
    if len(cluster_summary_df) > 0:
        poor_clusters = cluster_summary_df[cluster_summary_df['reachability_percentage'] < 80]
        if len(poor_clusters) > 0:
            print(f"🔍 {len(poor_clusters)} clusters have <80% reachability:")
            for _, cluster in poor_clusters.iterrows():
                print(f"   - Cluster {cluster['cluster_id']}: {cluster['reachability_percentage']:.1f}% reachable")
            print("   💡 Consider adding bus stops near these cluster centroids")
    
    # 2. Analyze overutilized stops
    stop_usage = assignments_df['assigned_stop_id'].value_counts()
    overutilized_stops = []
    
    for stop_id, usage_count in stop_usage.items():
        stop_info = assignment_system.bus_stops_data[
            assignment_system.bus_stops_data['cluster_id'] == stop_id
        ].iloc[0]
        utilization_rate = usage_count / stop_info['student_count']
        
        if utilization_rate > 0.9:  # Over 90% utilized
            overutilized_stops.append({
                'stop_id': stop_id,
                'usage': usage_count,
                'capacity': stop_info['student_count'],
                'utilization_rate': utilization_rate
            })
    
    if overutilized_stops:
        print(f"\n🚌 {len(overutilized_stops)} bus stops are over 90% utilized:")
        for stop in overutilized_stops:
            print(f"   - Stop {stop['stop_id']}: {stop['usage']}/{stop['capacity']} ({stop['utilization_rate']*100:.1f}%)")
        print("   💡 Consider increasing capacity or adding nearby stops")
    
    # 3. Analyze underutilized stops
    all_stops = set(assignment_system.bus_stops_data['cluster_id'])
    used_stops = set(assignments_df['assigned_stop_id'])
    unused_stops = all_stops - used_stops
    
    if unused_stops:
        print(f"\n🚏 {len(unused_stops)} bus stops are not being used:")
        print(f"   Stop IDs: {sorted(list(unused_stops))}")
        print("   💡 Consider relocating these stops to areas with higher student density")
    
    # 4. Distance analysis
    long_distance_students = assignments_df[assignments_df['road_distance_km'] > assignment_system.max_distance_km * 0.8]
    if len(long_distance_students) > 0:
        print(f"\n📏 {len(long_distance_students)} students have distances >80% of the limit:")
        avg_long_distance = long_distance_students['road_distance_km'].mean()
        print(f"   Average distance for these students: {avg_long_distance:.3f} km")
        print("   💡 Consider adding bus stops in areas with consistently long distances")
    
    # 5. Cluster size analysis
    if len(cluster_summary_df) > 0:
        small_clusters = cluster_summary_df[cluster_summary_df['student_count'] < 5]
        large_clusters = cluster_summary_df[cluster_summary_df['student_count'] > 30]
        
        if len(small_clusters) > 0:
            print(f"\n👥 {len(small_clusters)} clusters have <5 students:")
            print("   💡 These might benefit from merging with nearby clusters")
        
        if len(large_clusters) > 0:
            print(f"\n👥 {len(large_clusters)} clusters have >30 students:")
            print("   💡 Consider splitting these clusters or adding multiple stops")
    
    # 6. Coverage analysis
    total_students = len(assignment_system.student_data)
    assigned_students = len(assignments_df)
    coverage_rate = assigned_students / total_students * 100
    
    print(f"\n📈 COVERAGE ANALYSIS:")
    print(f"   Overall coverage: {coverage_rate:.1f}% of students assigned")
    
    if coverage_rate < 95:
        unassigned = total_students - assigned_students
        print(f"   {unassigned} students could not be assigned")
        print("   💡 Consider increasing distance limit or adding more bus stops")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Review the generated map to identify spatial patterns")
    print("2. Consider the recommendations above for infrastructure improvements")
    print("3. Validate assignments with actual road conditions and traffic")
    print("4. Test with different clustering parameters if needed")
    print("5. Monitor API usage and costs for large-scale operations")

def validate_api_key():
    """
    Validate Google Maps API key functionality
    """
    import googlemaps
    
    api_key = input("Please enter your Google Maps API key: ").strip()
    
    if not api_key or api_key == "YOUR_GOOGLE_MAPS_API_KEY_HERE":
        print("❌ Invalid API key. Please provide a valid Google Maps API key.")
        print("💡 Get your API key from: https://developers.google.com/maps/documentation/distance-matrix/get-api-key")
        return None
    
    try:
        gmaps = googlemaps.Client(key=api_key)
        
        # Test with a simple distance matrix call
        test_result = gmaps.distance_matrix(
            origins=[(40.7128, -74.0060)],  # New York
            destinations=[(40.7589, -73.9851)],  # Times Square
            mode="driving",
            units="metric"
        )
        
        if test_result['status'] == 'OK':
            print("✅ Google Maps API key is valid and working!")
            return api_key
        else:
            print("❌ API key validation failed. Please check your key and billing settings.")
            return None
            
    except Exception as e:
        print(f"❌ API validation error: {e}")
        print("Please check your API key and internet connection.")
        return None

# Alternative main function with API key validation
def main_with_validation():
    """
    Main function with interactive API key validation
    """
    print("🚌 Optimized Boarding Point Assignment System")
    print("=" * 50)
    
    # Validate API key first
    api_key = validate_api_key()
    if not api_key:
        return
    
    # Configuration
    MAX_DISTANCE_KM = 1.0
    MIN_CLUSTER_SIZE = 8
    MIN_SAMPLES = 5
    
    try:
        # Initialize the optimized assignment system
        print("\nInitializing system...")
        assignment_system = OptimizedBoardingPointAssignment(
            student_data_path="cleaned_student_locations_20250731_010007.csv",
            bus_stops_path="bus_stops_20250731_013718.csv",
            google_maps_api_key=api_key,
            max_distance_km=MAX_DISTANCE_KM
        )
        
        # Run the optimized assignment
        print("\nRunning optimized assignment...")
        assignments_df, cluster_summary_df = assignment_system.assign_boarding_points_optimized(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES
        )
        
        if len(assignments_df) > 0:
            # Generate all outputs
            assignment_system.generate_comprehensive_report(assignments_df, cluster_summary_df)
            assignment_system.create_enhanced_visualization(assignments_df, cluster_summary_df)
            generate_optimization_recommendations(assignments_df, cluster_summary_df, assignment_system)
            
            print(f"\n✅ Successfully processed {len(assignments_df)} student assignments!")
            
        else:
            print("❌ No assignments were made. Check your data and constraints.")
            
    except Exception as e:
        print(f"❌ Execution error: {e}")

if __name__ == "__main__":
    # Choose which main function to run
    main()  # Use this if you want to hardcode the API key
    #main_with_validation()  # Use this for interactive API key validation