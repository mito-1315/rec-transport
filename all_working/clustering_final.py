import pandas as pd
import numpy as np
import googlemaps
import json
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import folium
from folium import plugins
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import glob
from pathlib import Path

class EfficientStudentClusteringSystem:
    def __init__(self, google_maps_api_key, max_distance_km=1.0, api_call_limit=1000):
        """
        Initialize the efficient student clustering system
        
        Args:
            google_maps_api_key: Your Google Maps API key
            max_distance_km: Maximum road distance in km (default 1.0km)
            api_call_limit: Maximum API calls to use (default 100 for free tier)
        """
        self.max_distance_km = max_distance_km
        self.api_call_limit = api_call_limit
        self.api_calls_made = 0
        
        # Initialize Google Maps client
        self.gmaps = googlemaps.Client(key=google_maps_api_key)
        
        # Cache for distance calculations
        self.distance_cache = {}
        
        print(f"🚌 Student Clustering System Initialized")
        print(f"📏 Maximum distance constraint: {self.max_distance_km} km")
        print(f"🔑 API call limit: {self.api_call_limit}")
        
    def load_student_data(self, csv_path):
        """Load and validate student data"""
        print(f"📂 Loading student data from: {csv_path}")
        
        self.student_data = pd.read_csv(csv_path)
        
        # Validate required columns
        required_cols = ['latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in self.student_data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean data
        self.student_data = self.student_data.dropna(subset=['latitude', 'longitude'])
        self.student_data['latitude'] = pd.to_numeric(self.student_data['latitude'], errors='coerce')
        self.student_data['longitude'] = pd.to_numeric(self.student_data['longitude'], errors='coerce')
        self.student_data = self.student_data.dropna(subset=['latitude', 'longitude'])
        
        print(f"✅ Loaded {len(self.student_data)} valid student locations")
        return self.student_data
    
    def calculate_straight_line_distance(self, lat1, lon1, lat2, lon2):
        """Calculate straight-line distance using geodesic distance"""
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    
    def calculate_road_distance_conservative(self, origins, destinations, max_pairs=50):
        """
        Calculate road distances conservatively to preserve API calls
        
        Args:
            origins: List of (lat, lon) tuples
            destinations: List of (lat, lon) tuples  
            max_pairs: Maximum pairs to process in this call
            
        Returns:
            Dictionary with distances in km
        """
        if self.api_calls_made >= self.api_call_limit:
            print(f"⚠️ API call limit reached ({self.api_call_limit}). Using straight-line approximation.")
            return self._fallback_to_straight_line(origins, destinations)
        
        distances = {}
        pairs_processed = 0
        
        # Pre-filter by straight-line distance to reduce API calls
        valid_pairs = []
        for i, origin in enumerate(origins):
            for j, dest in enumerate(destinations):
                straight_dist = self.calculate_straight_line_distance(
                    origin[0], origin[1], dest[0], dest[1]
                )
                # Only check road distance if straight-line is reasonable  
                if straight_dist <= self.max_distance_km * 2.0:  # More generous buffer for Chennai traffic
                    valid_pairs.append((i, j, origin, dest))
        
        print(f"🔍 Pre-filtered to {len(valid_pairs)} valid pairs from {len(origins)*len(destinations)} total")
        
        # Process in small batches
        batch_size = min(10, max_pairs // len(valid_pairs) + 1) if valid_pairs else 10
        
        for batch_start in range(0, min(len(valid_pairs), max_pairs), batch_size):
            if self.api_calls_made >= self.api_call_limit:
                break
                
            batch_end = min(batch_start + batch_size, len(valid_pairs), max_pairs)
            batch_pairs = valid_pairs[batch_start:batch_end]
            
            batch_origins = [pair[2] for pair in batch_pairs]
            batch_destinations = [pair[3] for pair in batch_pairs]
            
            try:
                # Check cache first
                uncached_origins = []
                uncached_destinations = []
                cache_mapping = {}
                
                for i, (orig, dest) in enumerate(zip(batch_origins, batch_destinations)):
                    cache_key = f"{orig[0]:.6f},{orig[1]:.6f}|{dest[0]:.6f},{dest[1]:.6f}"
                    
                    if cache_key not in self.distance_cache:
                        if orig not in uncached_origins:
                            uncached_origins.append(orig)
                        if dest not in uncached_destinations:
                            uncached_destinations.append(dest)
                        cache_mapping[cache_key] = (len(uncached_origins)-1, len(uncached_destinations)-1)
                
                if uncached_origins and uncached_destinations:
                    # Make API call
                    matrix = self.gmaps.distance_matrix(
                        origins=uncached_origins,
                        destinations=uncached_destinations,
                        mode="driving",
                        units="metric",
                        avoid="tolls"
                    )
                    
                    self.api_calls_made += 1
                    print(f"🌐 API Call {self.api_calls_made}/{self.api_call_limit}")
                    
                    # Cache results
                    if matrix['status'] == 'OK':
                        for orig_idx, origin in enumerate(uncached_origins):
                            for dest_idx, destination in enumerate(uncached_destinations):
                                element = matrix['rows'][orig_idx]['elements'][dest_idx]
                                cache_key = f"{origin[0]:.6f},{origin[1]:.6f}|{destination[0]:.6f},{destination[1]:.6f}"
                                
                                if element['status'] == 'OK':
                                    distance_km = element['distance']['value'] / 1000.0
                                    self.distance_cache[cache_key] = distance_km
                                else:
                                    # Fallback to straight-line distance
                                    straight_dist = self.calculate_straight_line_distance(
                                        origin[0], origin[1], destination[0], destination[1]
                                    )
                                    self.distance_cache[cache_key] = straight_dist * 1.3  # Road factor approximation
                
                # Retrieve distances from cache
                for orig_idx, dest_idx, origin, dest in batch_pairs:
                    cache_key = f"{origin[0]:.6f},{origin[1]:.6f}|{dest[0]:.6f},{dest[1]:.6f}"
                    if cache_key in self.distance_cache:
                        distances[(orig_idx, dest_idx)] = self.distance_cache[cache_key]
                        pairs_processed += 1
                
                # Rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                print(f"❌ API Error: {e}")
                # Fallback to straight-line for this batch
                for orig_idx, dest_idx, origin, dest in batch_pairs:
                    straight_dist = self.calculate_straight_line_distance(
                        origin[0], origin[1], dest[0], dest[1]
                    )
                    distances[(orig_idx, dest_idx)] = straight_dist * 1.3
                    pairs_processed += 1
        
        print(f"✅ Processed {pairs_processed} distance calculations")
        return distances
    
    def _fallback_to_straight_line(self, origins, destinations):
        """Fallback to straight-line distance approximation"""
        distances = {}
        for i, origin in enumerate(origins):
            for j, dest in enumerate(destinations):
                straight_dist = self.calculate_straight_line_distance(
                    origin[0], origin[1], dest[0], dest[1]
                )
                # Apply road factor approximation (roads are typically 1.2-1.4x straight-line)
                distances[(i, j)] = straight_dist * 1.3
        return distances
    
    def create_initial_clusters_dbscan(self, eps_km=2.0, min_samples=3):
        """
        Create initial clusters using DBSCAN with geographic distance
        
        Args:
            eps_km: Maximum distance in km between points in same cluster
            min_samples: Minimum samples to form a dense region
            
        Returns:
            Cluster labels and cluster information
        """
        print("🔍 Creating initial geographic clusters using DBSCAN...")
        
        # Prepare coordinate data
        student_coords = self.student_data[['latitude', 'longitude']].astype(float).values
        
        # Convert km to degrees (rough approximation: 1 degree ≈ 111 km)
        eps_degrees = eps_km / 111.0
        
        print(f"   Using eps={eps_degrees:.6f} degrees (~{eps_km} km)")
        print(f"   Minimum samples per cluster: {min_samples}")
        
        # Apply DBSCAN clustering
        clusterer = DBSCAN(
            eps=eps_degrees,
            min_samples=min_samples,
            metric='euclidean'  # Works well with lat/lon for small distances
        )
        
        cluster_labels = clusterer.fit_predict(student_coords)
        
        # Analyze clusters
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"📊 Initial clustering results:")
        print(f"   - Number of clusters: {n_clusters}")
        print(f"   - Number of noise points: {n_noise}")
        print(f"   - Average cluster size: {(len(cluster_labels) - n_noise) / max(n_clusters, 1):.1f}")
        
        # If too many noise points, try with larger eps
        if n_noise > len(cluster_labels) * 0.5:  # More than 50% noise
            print("⚠️ Too many noise points, trying with larger distance...")
            eps_km_larger = eps_km * 1.5
            eps_degrees_larger = eps_km_larger / 111.0
            
            clusterer = DBSCAN(
                eps=eps_degrees_larger,
                min_samples=max(2, min_samples - 1),
                metric='euclidean'
            )
            
            cluster_labels = clusterer.fit_predict(student_coords)
            
            unique_labels = set(cluster_labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            print(f"📊 Adjusted clustering results:")
            print(f"   - Number of clusters: {n_clusters}")
            print(f"   - Number of noise points: {n_noise}")
            print(f"   - Average cluster size: {(len(cluster_labels) - n_noise) / max(n_clusters, 1):.1f}")
        
        self.student_data['initial_cluster'] = cluster_labels
        return cluster_labels, clusterer
        
    def create_agglomerative_clusters(self, max_clusters=50, distance_threshold_km=3.0):
        """
        Create clusters using agglomerative clustering as fallback for very spread data
        
        Args:
            max_clusters: Maximum number of clusters to create
            distance_threshold_km: Maximum distance between points in same cluster
            
        Returns:
            Cluster labels and cluster information
        """
        print("🔍 Creating clusters using Agglomerative Clustering...")
        
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import haversine_distances
        
        # Prepare coordinate data
        student_coords = self.student_data[['latitude', 'longitude']].astype(float).values
        student_coords_rad = np.radians(student_coords)
        
        # Calculate distance matrix
        distance_matrix = haversine_distances(student_coords_rad) * 6371  # Earth radius in km
        
        # Apply agglomerative clustering
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold_km,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clusterer.fit_predict(distance_matrix)
        
        # Analyze clusters
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels)
        
        # Calculate cluster sizes
        cluster_sizes = []
        for label in unique_labels:
            size = list(cluster_labels).count(label)
            cluster_sizes.append(size)
        
        print(f"📊 Agglomerative clustering results:")
        print(f"   - Number of clusters: {n_clusters}")
        print(f"   - Average cluster size: {np.mean(cluster_sizes):.1f}")
        print(f"   - Cluster size range: {min(cluster_sizes)}-{max(cluster_sizes)}")
        
        self.student_data['initial_cluster'] = cluster_labels
        return cluster_labels, clusterer
        
    def create_chennai_optimized_clusters(self):
        """
        Create clusters specifically optimized for Chennai's geographic spread
        Uses a multi-stage approach for the large metropolitan area
        """
        print("🏙️ Creating Chennai-optimized clusters...")
        
        from sklearn.cluster import KMeans
        
        # Prepare coordinate data
        student_coords = self.student_data[['latitude', 'longitude']].astype(float).values
        
        # For Chennai's scale, use K-means with geographic constraints
        # Estimate good number of clusters based on student density
        total_students = len(student_coords)
        
        # Target 20-50 students per cluster
        target_cluster_size = 10
        n_clusters = max(10, min(400, total_students // target_cluster_size))
        
        print(f"   Targeting {n_clusters} clusters for {total_students} students")
        print(f"   Expected cluster size: ~{total_students/n_clusters:.1f} students per cluster")
        
        # Apply K-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        cluster_labels = kmeans.fit_predict(student_coords)
        
        print(f"📊 K-means clustering results:")
        print(f"   - Number of clusters: {n_clusters}")
        print(f"   - No noise points (K-means assigns all points)")
        
        # Analyze cluster spread
        cluster_stats = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_students = student_coords[cluster_mask]
            
            if len(cluster_students) > 0:
                # Calculate cluster radius (max distance from centroid)
                centroid = cluster_students.mean(axis=0)
                max_dist = 0
                for student in cluster_students:
                    dist = self.calculate_straight_line_distance(
                        centroid[0], centroid[1], student[0], student[1]
                    )
                    max_dist = max(max_dist, dist)
                
                cluster_stats.append({
                    'cluster_id': cluster_id,
                    'size': len(cluster_students),
                    'radius_km': max_dist
                })
        
        # Show cluster size distribution
        sizes = [stat['size'] for stat in cluster_stats]
        radii = [stat['radius_km'] for stat in cluster_stats]
        
        print(f"   - Cluster sizes: {min(sizes)}-{max(sizes)} (avg: {np.mean(sizes):.1f})")
        print(f"   - Cluster radii: {min(radii):.1f}-{max(radii):.1f} km (avg: {np.mean(radii):.1f} km)")
        
        self.student_data['initial_cluster'] = cluster_labels
        
        return cluster_labels, kmeans
    
    def validate_and_split_clusters(self, cluster_labels):
        """
        Validate clusters using road distance and split if necessary
        """
        print("🛣️ Validating clusters with road distance constraints...")
        
        validated_clusters = []
        final_assignments = []
        
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        cluster_id_counter = 0
        
        for original_cluster_id in tqdm(unique_labels, desc="Validating clusters"):
            cluster_mask = cluster_labels == original_cluster_id
            cluster_students = self.student_data[cluster_mask].copy()
            
            if len(cluster_students) == 0:
                continue
            
            print(f"\n🔍 Processing cluster {original_cluster_id} with {len(cluster_students)} students")
            
            # Calculate cluster centroid
            centroid_lat = cluster_students['latitude'].mean()
            centroid_lon = cluster_students['longitude'].mean()
            
            # Get student coordinates
            student_coords = [(float(row['latitude']), float(row['longitude'])) 
                             for _, row in cluster_students.iterrows()]
            
            # Calculate road distances from centroid to all students in cluster
            centroid_coords = [(centroid_lat, centroid_lon)]
            
            distances = self.calculate_road_distance_conservative(
                student_coords, centroid_coords, max_pairs=min(50, len(student_coords))
            )
            
            # Check which students are within distance limit
            valid_students = []
            invalid_students = []
            
            for student_idx, (_, student_row) in enumerate(cluster_students.iterrows()):
                distance_key = (student_idx, 0)
                road_distance = distances.get(distance_key, float('inf'))
                
                if road_distance <= self.max_distance_km:
                    valid_students.append((student_row, road_distance))
                else:
                    invalid_students.append((student_row, road_distance))
            
            reachability = len(valid_students) / len(cluster_students) * 100
            print(f"   ✅ Reachability: {reachability:.1f}% ({len(valid_students)}/{len(cluster_students)})")
            
            if len(valid_students) >= 2:  # Minimum viable cluster size
                # Create validated cluster
                cluster_info = {
                    'cluster_id': cluster_id_counter,
                    'centroid_lat': centroid_lat,
                    'centroid_lon': centroid_lon,
                    'student_count': len(valid_students),
                    'avg_distance': np.mean([dist for _, dist in valid_students]),
                    'max_distance': max([dist for _, dist in valid_students]),
                    'reachability_percentage': reachability
                }
                
                validated_clusters.append(cluster_info)
                
                # Assign students to this cluster
                for student_row, distance in valid_students:
                    assignment = {
                        'user': student_row.get('ID', student_row.get('Roll Number', 'Unknown')),
                        'email': student_row.get('Email', 'N/A'),
                        'department': student_row.get('Department', 'N/A'),
                        'student_lat': float(student_row['latitude']),
                        'student_lon': float(student_row['longitude']),
                        'cluster_id': cluster_id_counter,
                        'centroid_lat': centroid_lat,
                        'centroid_lon': centroid_lon,
                        'road_distance_km': distance,
                        'within_distance_limit': True
                    }
                    final_assignments.append(assignment)
                
                cluster_id_counter += 1
            
            # Handle invalid students (try to create smaller sub-clusters or assign individually)
            if invalid_students:
                print(f"   ⚠️ {len(invalid_students)} students beyond distance limit")
                # For now, we'll handle them as individual clusters or merge with nearby clusters
                # This could be enhanced with sub-clustering logic
        
        # Handle noise points from original clustering
        noise_mask = cluster_labels == -1
        noise_students = self.student_data[noise_mask]
        
        if len(noise_students) > 0:
            print(f"\n🔍 Processing {len(noise_students)} noise points individually...")
            
            for _, student in noise_students.iterrows():
                # For noise points, create individual "clusters" or try to merge with nearby validated clusters
                # This is a simplified approach - could be enhanced
                assignment = {
                    'user': student.get('ID', student.get('Roll Number', 'Unknown')),
                    'email': student.get('Email', 'N/A'),
                    'department': student.get('Department', 'N/A'),
                    'student_lat': float(student['latitude']),
                    'student_lon': float(student['longitude']),
                    'cluster_id': cluster_id_counter,
                    'centroid_lat': float(student['latitude']),
                    'centroid_lon': float(student['longitude']),
                    'road_distance_km': 0.0,
                    'within_distance_limit': True
                }
                final_assignments.append(assignment)
                
                cluster_info = {
                    'cluster_id': cluster_id_counter,
                    'centroid_lat': float(student['latitude']),
                    'centroid_lon': float(student['longitude']),
                    'student_count': 1,
                    'avg_distance': 0.0,
                    'max_distance': 0.0,
                    'reachability_percentage': 100.0
                }
                validated_clusters.append(cluster_info)
                
                cluster_id_counter += 1
        
        assignments_df = pd.DataFrame(final_assignments)
        clusters_df = pd.DataFrame(validated_clusters)
        
        return assignments_df, clusters_df
    
    def create_visualization(self, assignments_df, clusters_df, output_path="student_clusters_map.html"):
        """
        Create an interactive map visualization
        """
        if len(assignments_df) == 0:
            print("❌ No assignments to visualize")
            return
        
        print(f"🗺️ Creating interactive map visualization...")
        
        # Calculate map center
        center_lat = assignments_df['student_lat'].mean()
        center_lon = assignments_df['student_lon'].mean()
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        # Color palette for clusters
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
                 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white',
                 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
        
        # Create color mapping
        unique_clusters = assignments_df['cluster_id'].unique()
        cluster_colors = {cluster_id: colors[i % len(colors)] for i, cluster_id in enumerate(unique_clusters)}
        
        # Add cluster centroids
        for _, cluster in clusters_df.iterrows():
            folium.Marker(
                location=[cluster['centroid_lat'], cluster['centroid_lon']],
                popup=folium.Popup(f"""
                    <b>Cluster {cluster['cluster_id']}</b><br>
                    Students: {cluster['student_count']}<br>
                    Avg Distance: {cluster['avg_distance']:.3f} km<br>
                    Max Distance: {cluster['max_distance']:.3f} km<br>
                    Reachability: {cluster['reachability_percentage']:.1f}%
                """, max_width=250),
                icon=folium.Icon(color='darkblue', icon='info-sign')
            ).add_to(m)
        
        # Add students with connections to centroids
        for _, assignment in assignments_df.iterrows():
            cluster_color = cluster_colors.get(assignment['cluster_id'], 'gray')

            # Compute distances to centroid
            straight_km = self.calculate_straight_line_distance(
                float(assignment['student_lat']),
                float(assignment['student_lon']),
                float(assignment['centroid_lat']),
                float(assignment['centroid_lon'])
            )

            road_km_value = assignment.get('road_distance_km', np.nan)
            try:
                road_km = float(road_km_value)
            except Exception:
                road_km = np.nan
            if pd.isna(road_km) or road_km <= 0:
                # Fallback approximation if road distance missing (e.g., noise/self-cluster)
                road_km = straight_km * 1.3

            # Student marker
            folium.CircleMarker(
                location=[assignment['student_lat'], assignment['student_lon']],
                radius=4,
                popup=folium.Popup(
                    f"""
                    <b>Student {assignment['user']}</b><br>
                    Department: {assignment['department']}<br>
                    Cluster: {assignment['cluster_id']}<br>
                    Road distance to centroid: {road_km:.3f} km<br>
                    Straight-line distance: {straight_km:.3f} km<br>
                    """,
                    max_width=240
                ),
                color=cluster_color,
                fill=True,
                fillColor=cluster_color,
                fillOpacity=0.7
            ).add_to(m)
            
            # Connection line to centroid
            folium.PolyLine(
                locations=[
                    [assignment['student_lat'], assignment['student_lon']],
                    [assignment['centroid_lat'], assignment['centroid_lon']]
                ],
                color='green',
                weight=1,
                opacity=0.4
            ).add_to(m)
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 250px; min-height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px">
        <p><b>Student Clustering Results</b></p>
        <p><i class="fa fa-map-marker" style="color:darkblue"></i> Cluster Centroids</p>
        <p><i class="fa fa-circle" style="color:red"></i> Students (colored by cluster)</p>
        <p><i class="fa fa-minus" style="color:green"></i> Student-Centroid Connections</p>
        <p>Max Distance: {self.max_distance_km} km</p>
        <p>Total Clusters: {len(clusters_df)}</p>
        <p>API Calls Used: {self.api_calls_made}/{self.api_call_limit}</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        m.save(output_path)
        print(f"✅ Map saved to: {output_path}")
    
    def generate_report(self, assignments_df, clusters_df, output_dir=".", file_prefix="clusters"):
        """
        Generate comprehensive clustering report and save in specified directory
        
        Args:
            assignments_df: DataFrame with student assignments
            clusters_df: DataFrame with cluster information
            output_dir: Directory to save output files
            file_prefix: Prefix for output files
        """
        if len(assignments_df) == 0:
            print("❌ No assignments to report")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output file paths
        assignments_path = os.path.join(output_dir, f"{file_prefix}_assignments.csv")
        centroids_path = os.path.join(output_dir, f"{file_prefix}_centroids.csv")
        map_path = os.path.join(output_dir, f"{file_prefix}_map.html")
        
        # Save detailed results
        assignments_df.to_csv(assignments_path, index=False)
        clusters_df.to_csv(centroids_path, index=False)
        
        # Create visualization
        self.create_visualization(assignments_df, clusters_df, map_path)
        
        # Calculate statistics
        total_students = len(self.student_data)
        clustered_students = len(assignments_df)
        total_clusters = len(clusters_df)
        
        avg_distance = assignments_df['road_distance_km'].mean()
        avg_cluster_size = clusters_df['student_count'].mean()
        avg_reachability = clusters_df['reachability_percentage'].mean()
        
        print("\n" + "="*60)
        print("🚌 STUDENT CLUSTERING REPORT")
        print("="*60)
        print(f"📊 SUMMARY:")
        print(f"   Total students: {total_students}")
        print(f"   Students clustered: {clustered_students} ({clustered_students/total_students*100:.1f}%)")
        print(f"   Total clusters: {total_clusters}")
        print(f"   Average cluster size: {avg_cluster_size:.1f} students")
        print()
        print(f"📏 DISTANCE STATISTICS:")
        print(f"   Average distance to centroid: {avg_distance:.3f} km")
        print(f"   Maximum distance constraint: {self.max_distance_km} km")
        print(f"   Average reachability: {avg_reachability:.1f}%")
        print()
        print(f"🌐 API USAGE:")
        print(f"   API calls used: {self.api_calls_made}/{self.api_call_limit}")
        print(f"   Efficiency: {clustered_students/max(self.api_calls_made, 1):.1f} students per API call")
        print()
        
        # Cluster size distribution
        print("📈 CLUSTER SIZE DISTRIBUTION:")
        size_bins = [0, 5, 10, 20, 30, 50, float('inf')]
        for i in range(len(size_bins)-1):
            if size_bins[i+1] == float('inf'):
                count = len(clusters_df[clusters_df['student_count'] > size_bins[i]])
                label = f"   > {size_bins[i]} students"
            else:
                count = len(clusters_df[
                    (clusters_df['student_count'] > size_bins[i]) & 
                    (clusters_df['student_count'] <= size_bins[i+1])
                ])
                label = f"   {size_bins[i]+1}-{size_bins[i+1]} students"
            
            percentage = count / total_clusters * 100 if total_clusters > 0 else 0
            print(f"{label}: {count} clusters ({percentage:.1f}%)")
        
        print()
        print("📁 FILES GENERATED:")
        print(f"   - {assignments_path} (detailed student assignments)")
        print(f"   - {centroids_path} (cluster centroid information)")
        print(f"   - {map_path} (interactive map visualization)")
        print("="*60)

    def reset_for_new_file(self):
        """Reset system state for processing a new file"""
        self.api_calls_made = 0
        self.distance_cache = {}
        self.student_data = None

def find_csv_files(base_directory):
    """
    Recursively find all CSV files in the directory structure, excluding leave.csv files
    
    Args:
        base_directory: Base directory to search (e.g., "Routes_Data")
    
    Returns:
        List of tuples: (full_path, relative_path, output_directory, file_prefix)
    """
    csv_files = []
    
    # Use glob to find all CSV files recursively
    pattern = os.path.join(base_directory, "**", "*.csv")
    
    for csv_path in glob.glob(pattern, recursive=True):
        # Get the relative path from base directory
        rel_path = os.path.relpath(csv_path, base_directory)
        
        # Skip leave.csv files and coords.csv (bus stops)
        filename = os.path.basename(csv_path).lower()
        if filename in ['leave.csv', 'coords.csv']:
            print(f"⏭️ Skipping: {rel_path}")
            continue
        
        # Extract directory and filename components
        dir_path = os.path.dirname(csv_path)
        file_prefix = os.path.splitext(os.path.basename(csv_path))[0]  # Remove .csv extension
        
        csv_files.append((csv_path, rel_path, dir_path, file_prefix))
    
    return csv_files

def process_multiple_files(base_directory="Routes_Data_5800", google_maps_api_key="YOUR_API_KEY", 
                         max_distance_km=1.0, total_api_budget=200):
    """
    Process multiple CSV files until free tier budget is exhausted, excluding leave.csv files
    
    Args:
        base_directory: Base directory containing CSV files
        google_maps_api_key: Your Google Maps API key
        max_distance_km: Maximum distance constraint in km
        total_api_budget: Total API budget in dollars (default $200 for free tier)
    """
    print("🚀 Starting Multi-File Student Clustering System (Free Tier Mode)")
    print("="*70)
    
    # Find all CSV files (excluding leave.csv and coords.csv)
    csv_files = find_csv_files(base_directory)
    
    if not csv_files:
        print(f"❌ No valid CSV files found in {base_directory}")
        return
    
    print(f"📁 Found {len(csv_files)} valid CSV files to process:")
    for csv_path, rel_path, _, _ in csv_files:
        print(f"   - {rel_path}")
    print(f"\n🚫 Excluded files: leave.csv, coords.csv")
    print(f"💰 API Budget: ${total_api_budget} (Free Tier)")
    print()
    
    # Cost tracking
    api_cost_per_1000_elements = 5.0  # $5 per 1000 elements
    total_cost_spent = 0.0
    total_elements_used = 0
    total_api_calls_made = 0
    
    # Initialize clustering system with dynamic limits
    api_call_limit_per_file = 500  # Conservative starting limit
    
    clustering_system = EfficientStudentClusteringSystem(
        google_maps_api_key=google_maps_api_key,
        max_distance_km=max_distance_km,
        api_call_limit=api_call_limit_per_file
    )
    
    # Process each file until budget exhausted
    successful_files = 0
    failed_files = []
    budget_exhausted = False
    
    for i, (csv_path, rel_path, output_dir, file_prefix) in enumerate(csv_files, 1):
        print(f"\n{'='*70}")
        print(f"🔄 Processing file {i}/{len(csv_files)}: {rel_path}")
        print(f"💰 Budget remaining: ${total_api_budget - total_cost_spent:.2f}")
        print(f"{'='*70}")
        
        # Check if we have budget remaining
        if total_cost_spent >= total_api_budget:
            print(f"🛑 FREE TIER BUDGET EXHAUSTED!")
            print(f"   Total spent: ${total_cost_spent:.2f}")
            print(f"   Stopping processing to avoid charges...")
            budget_exhausted = True
            break
        
        try:
            # Reset system for new file
            clustering_system.reset_for_new_file()
            clustering_system.api_call_limit = api_call_limit_per_file
            
            # Load student data
            student_data = clustering_system.load_student_data(csv_path)
            
            if len(student_data) == 0:
                print(f"⚠️ No valid student data in {rel_path}, skipping...")
                continue
            
            # Estimate cost before processing
            total_students = len(student_data)
            estimated_clusters = max(1, total_students // 25)  # Rough estimate
            estimated_elements = estimated_clusters * 20  # Conservative estimate
            estimated_cost = (estimated_elements / 1000.0) * api_cost_per_1000_elements
            
            print(f"📊 Pre-processing estimates:")
            print(f"   - Students: {total_students}")
            print(f"   - Estimated clusters: {estimated_clusters}")
            print(f"   - Estimated API cost: ${estimated_cost:.2f}")
            
            # Check if we can afford this file
            if total_cost_spent + estimated_cost > total_api_budget:
                print(f"⚠️ Estimated cost (${estimated_cost:.2f}) would exceed remaining budget")
                print(f"   Remaining budget: ${total_api_budget - total_cost_spent:.2f}")
                print(f"🛑 Stopping to stay within free tier...")
                budget_exhausted = True
                break
            
            # Choose clustering method based on data size and remaining budget
            if total_students > 500 and (total_api_budget - total_cost_spent) > 10:  # Need at least $10 buffer
                print("🏙️ Using Chennai-optimized clustering for large dataset...")
                cluster_labels, clusterer = clustering_system.create_chennai_optimized_clusters()
                
            elif total_students > 100:
                print("🔍 Using DBSCAN clustering for medium dataset...")
                cluster_labels, clusterer = clustering_system.create_initial_clusters_dbscan(
                    eps_km=3.0,
                    min_samples=2
                )
                
                # Check if DBSCAN worked well
                n_noise = list(cluster_labels).count(-1)
                noise_percentage = n_noise / len(cluster_labels) * 100
                
                if noise_percentage > 60:
                    print("⚠️ DBSCAN created too much noise, switching to Agglomerative...")
                    cluster_labels, clusterer = clustering_system.create_agglomerative_clusters(
                        distance_threshold_km=5.0
                    )
            
            else:  # Small dataset
                print("🔗 Using Agglomerative clustering for small dataset...")
                cluster_labels, clusterer = clustering_system.create_agglomerative_clusters(
                    distance_threshold_km=4.0
                )
            
            # Store API calls before validation
            api_calls_before = clustering_system.api_calls_made
            
            # Validate clusters with road distance constraints
            assignments_df, clusters_df = clustering_system.validate_and_split_clusters(cluster_labels)
            
            # Calculate actual cost for this file
            api_calls_this_file = clustering_system.api_calls_made
            # Estimate elements (conservative: 20 elements per API call on average)
            elements_this_file = api_calls_this_file * 15  # Conservative estimate
            cost_this_file = (elements_this_file / 1000.0) * api_cost_per_1000_elements
            
            # Update totals
            total_cost_spent += cost_this_file
            total_elements_used += elements_this_file
            total_api_calls_made += api_calls_this_file
            
            print(f"\n💰 Cost tracking for this file:")
            print(f"   - API calls made: {api_calls_this_file}")
            print(f"   - Elements used: ~{elements_this_file}")
            print(f"   - Cost: ${cost_this_file:.2f}")
            print(f"   - Total spent so far: ${total_cost_spent:.2f}")
            print(f"   - Budget remaining: ${total_api_budget - total_cost_spent:.2f}")
            
            # Check if we're approaching budget limit
            if total_cost_spent > total_api_budget * 0.9:  # 90% of budget used
                print(f"⚠️ WARNING: 90% of budget used! Processing carefully...")
                api_call_limit_per_file = max(10, api_call_limit_per_file // 2)  # Reduce limits
                print(f"   - Reducing API limit per file to: {api_call_limit_per_file}")
            
            if len(assignments_df) > 0:
                # Generate outputs in the same directory as the input file
                clustering_system.generate_report(
                    assignments_df, 
                    clusters_df, 
                    output_dir=output_dir,
                    file_prefix=file_prefix
                )
                
                print(f"\n✅ Successfully processed {rel_path}")
                print(f"   - Clustered {len(assignments_df)} students into {len(clusters_df)} clusters")
                print(f"   - Files saved in: {output_dir}")
                
                successful_files += 1
                
            else:
                print(f"❌ No valid clusters created for {rel_path}")
                failed_files.append((rel_path, "No valid clusters created"))
            
            # Safety check - if we're very close to budget, stop
            if total_cost_spent >= total_api_budget * 0.95:  # 95% of budget used
                print(f"\n🛑 APPROACHING BUDGET LIMIT!")
                print(f"   95% of free tier budget used (${total_cost_spent:.2f})")
                print(f"   Stopping to avoid any charges...")
                budget_exhausted = True
                break
                
        except Exception as e:
            print(f"❌ Error processing {rel_path}: {e}")
            failed_files.append((rel_path, str(e)))
            continue
    
    # Final summary
    print(f"\n{'='*70}")
    print("🏁 FREE TIER PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Successfully processed: {successful_files}/{len(csv_files)} files")
    print(f"💰 COST SUMMARY:")
    print(f"   - Total API calls made: {total_api_calls_made}")
    print(f"   - Total elements used: ~{total_elements_used:,}")
    print(f"   - Total cost: ${total_cost_spent:.2f}")
    print(f"   - Free tier budget: ${total_api_budget}")
    print(f"   - Remaining budget: ${total_api_budget - total_cost_spent:.2f}")
    print(f"   - Budget utilization: {(total_cost_spent/total_api_budget)*100:.1f}%")
    
    if budget_exhausted:
        remaining_files = len(csv_files) - i
        print(f"\n🛑 BUDGET LIMIT REACHED:")
        print(f"   - Processed: {successful_files} files")
        print(f"   - Remaining unprocessed: {remaining_files} files")
        print(f"   - To process remaining files, wait for next month's free credit")
        print(f"     or consider upgrading to paid plan")
    
    if failed_files:
        print(f"\n❌ Failed files:")
        for file_path, error in failed_files:
            print(f"   - {file_path}: {error}")
    
    print(f"\n💡 OPTIMIZATION SUGGESTIONS:")
    if total_cost_spent < total_api_budget * 0.5:
        print(f"   - You're using only {(total_cost_spent/total_api_budget)*100:.1f}% of free tier")
        print(f"   - Consider processing larger datasets or reducing distance constraints")
    elif total_cost_spent > total_api_budget * 0.8:
        print(f"   - You're using {(total_cost_spent/total_api_budget)*100:.1f}% of free tier")
        print(f"   - Consider increasing pre-filtering or processing less frequently")
    
    print(f"{'='*70}")

def main():
    """
    Main function to run the student clustering system with free tier limits
    """
    # Configuration
    GOOGLE_MAPS_API_KEY = "AIzaSyAiVn2TbI7qSuTzw1EKvY4urq7V5aTZkZg"  # Replace with your actual API key
    MAX_DISTANCE_KM = 1.0
    FREE_TIER_BUDGET = 200.0  # $200 monthly free credit
    BASE_DIRECTORY = "Routes_Data_5800"  # Base directory containing all CSV files
    
    try:
        # Process multiple files with budget constraints
        process_multiple_files(
            base_directory=BASE_DIRECTORY,
            google_maps_api_key=GOOGLE_MAPS_API_KEY,
            max_distance_km=MAX_DISTANCE_KM,
            total_api_budget=FREE_TIER_BUDGET
        )
            
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

def process_single_file(csv_path, google_maps_api_key="YOUR_API_KEY", max_distance_km=1.0, api_call_limit=100):
    """
    Process a single CSV file (for testing or specific use cases)
    
    Args:
        csv_path: Path to the CSV file
        google_maps_api_key: Your Google Maps API key
        max_distance_km: Maximum distance constraint
        api_call_limit: API call limit
    """
    print(f"🚀 Processing single file: {csv_path}")
    print("="*50)
    
    try:
        # Initialize system
        clustering_system = EfficientStudentClusteringSystem(
            google_maps_api_key=google_maps_api_key,
            max_distance_km=max_distance_km,
            api_call_limit=api_call_limit
        )
        
        # Load student data
        student_data = clustering_system.load_student_data(csv_path)
        
        # Choose clustering method
        total_students = len(student_data)
        
        if total_students > 500:
            cluster_labels, clusterer = clustering_system.create_chennai_optimized_clusters()
        elif total_students > 100:
            cluster_labels, clusterer = clustering_system.create_initial_clusters_dbscan(eps_km=3.0, min_samples=2)
        else:
            cluster_labels, clusterer = clustering_system.create_agglomerative_clusters(distance_threshold_km=4.0)
        
        # Validate clusters
        assignments_df, clusters_df = clustering_system.validate_and_split_clusters(cluster_labels)
        
        if len(assignments_df) > 0:
            # Get output directory and prefix
            output_dir = os.path.dirname(csv_path)
            file_prefix = os.path.splitext(os.path.basename(csv_path))[0]
            
            # Generate outputs
            clustering_system.generate_report(assignments_df, clusters_df, output_dir, file_prefix)
            
            print(f"\n✅ Successfully processed {csv_path}")
            print(f"🌐 Used {clustering_system.api_calls_made} API calls")
            
        else:
            print("❌ No valid clusters were created. Try adjusting parameters.")
            
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the multi-file processor by default
    main()
    
    # Uncomment below to process a single file instead:
    # process_single_file("Routes_Data/Friday/3_pm/3_pm.csv")