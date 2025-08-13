import pandas as pd
import numpy as np
import json
import os
from math import radians, cos, sin, asin, sqrt
from sklearn.neighbors import BallTree
import time
from pathlib import Path

class BatchOptimalRouteSnapper:
    def __init__(self, outlier_threshold_meters=1000):
        self.route_points = []
        self.route_metadata = []
        self.ball_tree = None
        self.outlier_threshold = outlier_threshold_meters
        
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        Returns distance in meters
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radius of earth in meters
        r = 6371000
        return c * r
    
    def load_route_data(self, route_json_path):
        """
        Load route data from JSON file and extract all coordinate points
        """
        print("Loading route data...")
        start_time = time.time()
        
        with open(route_json_path, 'r') as f:
            route_data = json.load(f)
        
        self.route_points = []
        self.route_metadata = []
        
        # Process highways
        if 'highways' in route_data:
            for route in route_data['highways']:
                route_name = route.get('name', f"Highway_{route.get('id', 'unknown')}")
                route_type = 'highway'
                
                if 'coordinates' in route and route['coordinates']:
                    for coord in route['coordinates']:
                        self.route_points.append([coord[0], coord[1]])  # [lat, lon]
                        self.route_metadata.append({
                            'route_name': route_name,
                            'route_type': route_type,
                            'route_id': route.get('id', None)
                        })
        
        # Process arterials
        if 'arterials' in route_data:
            for route in route_data['arterials']:
                route_name = route.get('name', f"Arterial_{route.get('id', 'unknown')}")
                route_type = 'arterial'
                
                if 'coordinates' in route and route['coordinates']:
                    for coord in route['coordinates']:
                        self.route_points.append([coord[0], coord[1]])  # [lat, lon]
                        self.route_metadata.append({
                            'route_name': route_name,
                            'route_type': route_type,
                            'route_id': route.get('id', None)
                        })
        
        # Convert to numpy array for efficient processing
        self.route_points = np.array(self.route_points)
        
        # Build BallTree for ultra-fast nearest neighbor search
        # Convert to radians for BallTree
        route_points_rad = np.radians(self.route_points)
        self.ball_tree = BallTree(route_points_rad, metric='haversine')
        
        load_time = time.time() - start_time
        print(f"Loaded {len(self.route_points)} route points in {load_time:.2f} seconds")
        
    def load_cluster_data(self, cluster_csv_path):
        """
        Load cluster data from CSV file
        """
        print(f"Loading cluster data from {cluster_csv_path}...")
        
        # Read the CSV file
        df = pd.read_csv(cluster_csv_path)
        
        # Extract required columns
        clusters = []
        for _, row in df.iterrows():
            clusters.append({
                'cluster_number': int(row['cluster_id']),
                'original_lat': float(row['centroid_lat']),
                'original_lon': float(row['centroid_lon']),
                'num_students': int(row['student_count']),
            })
        
        print(f"Loaded {len(clusters)} clusters")
        return clusters
    
    def snap_clusters_to_routes(self, clusters):
        """
        Snap cluster centroids to nearest route points using optimized algorithm
        Returns normal results and outliers separately
        """
        if self.ball_tree is None:
            raise ValueError("Route data not loaded. Call load_route_data() first.")
        
        print("Snapping clusters to routes...")
        start_time = time.time()
        
        normal_results = []
        outlier_results = []
        
        for cluster in clusters:
            # Original cluster coordinates
            orig_lat = cluster['original_lat']
            orig_lon = cluster['original_lon']
            
            # Convert to radians for BallTree query
            cluster_point_rad = np.radians([[orig_lat, orig_lon]])
            
            # Find nearest route point using BallTree (ultra-fast)
            distances, indices = self.ball_tree.query(cluster_point_rad, k=1)
            
            # Get the nearest point index
            nearest_idx = indices[0][0]
            
            # Get snapped coordinates
            snapped_lat = self.route_points[nearest_idx][0]
            snapped_lon = self.route_points[nearest_idx][1]
            
            # Get route metadata
            route_info = self.route_metadata[nearest_idx]
            
            # Calculate actual distance using Haversine formula
            snap_distance = self.haversine_distance(
                orig_lat, orig_lon, snapped_lat, snapped_lon
            )
            
            # Store result
            result = {
                'cluster_number': cluster['cluster_number'],
                'original_lat': orig_lat,
                'original_lon': orig_lon,
                'snapped_lat': snapped_lat,
                'snapped_lon': snapped_lon,
                'route_name': route_info['route_name'],
                'route_type': route_info['route_type'],
                'route_id': route_info['route_id'],
                'snap_distance_meters': round(snap_distance, 2),
                'num_students': cluster['num_students'],
            }
            
            # Separate normal results from outliers
            if snap_distance > self.outlier_threshold:
                outlier_results.append(result)
            else:
                normal_results.append(result)
        
        snap_time = time.time() - start_time
        print(f"Snapped {len(clusters)} clusters ({len(normal_results)} normal, {len(outlier_results)} outliers) in {snap_time:.2f} seconds")
        
        return normal_results, outlier_results
    
    def save_results(self, results, output_path):
        """
        Save snapping results to CSV file
        """
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        else:
            print(f"No results to save to {output_path}")
    
    def save_outliers(self, outliers, output_path):
        """
        Save outlier results to CSV file
        """
        if outliers:
            df = pd.DataFrame(outliers)
            df.to_csv(output_path, index=False)
            print(f"Outliers (>{self.outlier_threshold}m snap distance) saved to {output_path}")
        else:
            print(f"No outliers found (no clusters snapped >{self.outlier_threshold}m)")
        
    def print_summary(self, normal_results, outliers, file_name):
        """
        Print summary statistics including outlier information
        """
        all_results = normal_results + outliers
        
        if not all_results:
            print("No results to summarize")
            return
        
        all_distances = [r['snap_distance_meters'] for r in all_results]
        normal_distances = [r['snap_distance_meters'] for r in normal_results]
        outlier_distances = [r['snap_distance_meters'] for r in outliers]
        
        print("\n" + "="*60)
        print(f"CLUSTER SNAPPING SUMMARY - {file_name}")
        print("="*60)
        print(f"Total clusters processed: {len(all_results)}")
        print(f"Normal clusters (<= {self.outlier_threshold}m): {len(normal_results)}")
        print(f"Outlier clusters (> {self.outlier_threshold}m): {len(outliers)}")
        print(f"Outlier percentage: {(len(outliers)/len(all_results)*100):.1f}%")
        
        print(f"\nOVERALL STATISTICS:")
        print(f"Average snap distance: {np.mean(all_distances):.2f} meters")
        print(f"Minimum snap distance: {np.min(all_distances):.2f} meters")
        print(f"Maximum snap distance: {np.max(all_distances):.2f} meters")
        print(f"Median snap distance: {np.median(all_distances):.2f} meters")
        
        if normal_distances:
            print(f"\nNORMAL CLUSTERS STATISTICS:")
            print(f"Average distance: {np.mean(normal_distances):.2f} meters")
            print(f"Maximum distance: {np.max(normal_distances):.2f} meters")
        
        if outlier_distances:
            print(f"\nOUTLIER CLUSTERS STATISTICS:")
            print(f"Average distance: {np.mean(outlier_distances):.2f} meters")
            print(f"Minimum distance: {np.min(outlier_distances):.2f} meters")
            print(f"Maximum distance: {np.max(outlier_distances):.2f} meters")
        
        # Route type distribution for all clusters
        route_types = {}
        for result in all_results:
            route_type = result['route_type']
            route_types[route_type] = route_types.get(route_type, 0) + 1
        
        print(f"\nAll clusters snapped by route type:")
        for route_type, count in route_types.items():
            print(f"  {route_type}: {count} clusters")
        
        # Route type distribution for outliers only
        if outliers:
            outlier_route_types = {}
            for result in outliers:
                route_type = result['route_type']
                outlier_route_types[route_type] = outlier_route_types.get(route_type, 0) + 1
            
            print(f"\nOutlier clusters by route type:")
            for route_type, count in outlier_route_types.items():
                print(f"  {route_type}: {count} outliers")
        
        print("\n" + "="*60)
    
    def find_centroid_files(self, base_path):
        """
        Find all centroid files in the directory structure
        """
        centroid_files = []
        base_path = Path(base_path)
        
        # Walk through all subdirectories
        for file_path in base_path.rglob("*_centroids.csv"):
            centroid_files.append(file_path)
        
        return sorted(centroid_files)
    
    def process_all_files(self, base_path, route_json_path):
        """
        Process all centroid files found in the directory structure
        """
        # Load route data once (it's the same for all files)
        self.load_route_data(route_json_path)
        
        # Find all centroid files
        centroid_files = self.find_centroid_files(base_path)
        
        if not centroid_files:
            print("No centroid files found!")
            return
        
        print(f"\nFound {len(centroid_files)} centroid files to process:")
        for file_path in centroid_files:
            print(f"  - {file_path}")
        
        print(f"\nOutlier threshold set to: {self.outlier_threshold} meters")
        print("\n" + "="*80)
        print("STARTING BATCH PROCESSING")
        print("="*80)
        
        processed_count = 0
        total_outliers = 0
        total_normal = 0
        total_time_start = time.time()
        
        for centroid_file in centroid_files:
            try:
                print(f"\nProcessing: {centroid_file.name}")
                print("-" * 60)
                
                # Load cluster data
                clusters = self.load_cluster_data(centroid_file)
                
                # Perform snapping (returns normal and outliers separately)
                normal_results, outlier_results = self.snap_clusters_to_routes(clusters)
                
                # Create output filenames in the same directory
                normal_output_file = centroid_file.parent / f"{centroid_file.stem}_snapped.csv"
                outlier_output_file = centroid_file.parent / f"{centroid_file.stem}_snap_outliers.csv"
                
                # Save results
                self.save_results(normal_results, normal_output_file)
                self.save_outliers(outlier_results, outlier_output_file)
                
                # Print summary
                self.print_summary(normal_results, outlier_results, centroid_file.name)
                
                processed_count += 1
                total_outliers += len(outlier_results)
                total_normal += len(normal_results)
                
            except Exception as e:
                print(f"Error processing {centroid_file}: {e}")
                continue
        
        total_time = time.time() - total_time_start
        total_clusters = total_normal + total_outliers
        
        print("\n" + "="*80)
        print("BATCH PROCESSING COMPLETE")
        print("="*80)
        print(f"Successfully processed: {processed_count}/{len(centroid_files)} files")
        print(f"Total clusters processed: {total_clusters}")
        print(f"Total normal clusters: {total_normal}")
        print(f"Total outlier clusters: {total_outliers}")
        if total_clusters > 0:
            print(f"Overall outlier percentage: {(total_outliers/total_clusters*100):.1f}%")
        print(f"Total processing time: {total_time:.2f} seconds")
        print("="*80)

def main():
    # Initialize the batch snapper with 1km outlier threshold
    snapper = BatchOptimalRouteSnapper(outlier_threshold_meters=1000)
    
    # File paths - Update these to match your setup
    base_path = "Routes_Data"  # Base directory containing all the day/time folders
    route_json_path = 'all_working/chennai-complete-roads-60km-precision-2025-08-06 copy.json'
    
    try:
        # Process all centroid files in the directory structure
        snapper.process_all_files(base_path, route_json_path)
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please ensure the route JSON file and base directory are in the correct location")
    except Exception as e:
        print(f"Error: {e}")

def process_specific_files():
    """
    Alternative function to process specific files if you want more control
    """
    snapper = BatchOptimalRouteSnapper(outlier_threshold_meters=1000)
    
    # Define specific file mappings
    files_to_process = [
        {
            'input': 'Routes_Data/Friday/3_pm/3_pm_centroids.csv',
            'normal_output': 'Routes_Data/Friday/3_pm/3_pm_centroids_snapped.csv',
            'outlier_output': 'Routes_Data/Friday/3_pm/3_pm_centroids_snap_outliers.csv'
        },
        {
            'input': 'Routes_Data/Friday/5_pm/5_pm_centroids.csv',
            'normal_output': 'Routes_Data/Friday/5_pm/5_pm_centroids_snapped.csv',
            'outlier_output': 'Routes_Data/Friday/5_pm/5_pm_centroids_snap_outliers.csv'
        },
        # Add more file mappings as needed
    ]
    
    route_json_path = 'all_working/chennai-complete-roads-60km-precision-2025-08-06 copy.json'
    # Load route data once
    snapper.load_route_data(route_json_path)
    
    for file_info in files_to_process:
        try:
            if os.path.exists(file_info['input']):
                print(f"\nProcessing: {file_info['input']}")
                
                # Load and process
                clusters = snapper.load_cluster_data(file_info['input'])
                normal_results, outlier_results = snapper.snap_clusters_to_routes(clusters)
                
                # Save results
                snapper.save_results(normal_results, file_info['normal_output'])
                snapper.save_outliers(outlier_results, file_info['outlier_output'])
                snapper.print_summary(normal_results, outlier_results, os.path.basename(file_info['input']))
                
            else:
                print(f"File not found: {file_info['input']}")
                
        except Exception as e:
            print(f"Error processing {file_info['input']}: {e}")

if __name__ == "__main__":
    # Run the main batch processing
    main()
    
    # Uncomment below to use the specific file processing instead
    # process_specific_files()