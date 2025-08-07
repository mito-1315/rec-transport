import pandas as pd
import numpy as np
import json
from math import radians, cos, sin, asin, sqrt
from sklearn.neighbors import BallTree
import time

class OptimalRouteSnapper:
    def __init__(self):
        self.route_points = []
        self.route_metadata = []
        self.ball_tree = None
        
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
        print("Loading cluster data...")
        
        # Read the CSV file
        df = pd.read_csv(cluster_csv_path)
        
        # Extract required columns
        clusters = []
        for _, row in df.iterrows():
            clusters.append({
                'cluster_number': row['cluster_number'],
                'original_lat': float(row['cluster_lat']),
                'original_lon': float(row['cluster_long']),
                'num_students': int(row['num_students']),
                'user_ids': row['user_ids']
            })
        
        print(f"Loaded {len(clusters)} clusters")
        return clusters
    
    def snap_clusters_to_routes(self, clusters):
        """
        Snap cluster centroids to nearest route points using optimized algorithm
        """
        if self.ball_tree is None:
            raise ValueError("Route data not loaded. Call load_route_data() first.")
        
        print("Snapping clusters to routes...")
        start_time = time.time()
        
        snapped_results = []
        
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
                'user_ids': cluster['user_ids']
            }
            
            snapped_results.append(result)
        
        snap_time = time.time() - start_time
        print(f"Snapped {len(snapped_results)} clusters in {snap_time:.2f} seconds")
        
        return snapped_results
    
    def save_results(self, results, output_path):
        """
        Save snapping results to CSV file
        """
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
    def print_summary(self, results):
        """
        Print summary statistics
        """
        if not results:
            print("No results to summarize")
            return
        
        distances = [r['snap_distance_meters'] for r in results]
        
        print("\n" + "="*60)
        print("CLUSTER SNAPPING SUMMARY")
        print("="*60)
        print(f"Total clusters processed: {len(results)}")
        print(f"Average snap distance: {np.mean(distances):.2f} meters")
        print(f"Minimum snap distance: {np.min(distances):.2f} meters")
        print(f"Maximum snap distance: {np.max(distances):.2f} meters")
        print(f"Median snap distance: {np.median(distances):.2f} meters")
        
        # Route type distribution
        route_types = {}
        for result in results:
            route_type = result['route_type']
            route_types[route_type] = route_types.get(route_type, 0) + 1
        
        print(f"\nClusters snapped by route type:")
        for route_type, count in route_types.items():
            print(f"  {route_type}: {count} clusters")
        
        print("\n" + "="*60)

def main():
    # Initialize the snapper
    snapper = OptimalRouteSnapper()
    
    # File paths (update these to match your file locations)
    route_json_path = 'chennai-complete-roads-60km-precision-2025-08-06.json'
    cluster_csv_path = 'clustered_students.csv'
    output_csv_path = 'snapped_clusters_results.csv'
    
    try:
        # Load route data
        snapper.load_route_data(route_json_path)
        
        # Load cluster data
        clusters = snapper.load_cluster_data(cluster_csv_path)
        
        # Perform optimal snapping
        results = snapper.snap_clusters_to_routes(clusters)
        
        # Print detailed results
        print("\nDETAILED RESULTS:")
        print("-" * 80)
        for result in results:
            print(f"Cluster {result['cluster_number']:2d} | "
                  f"Students: {result['num_students']:2d} | "
                  f"Distance: {result['snap_distance_meters']:6.1f}m | "
                  f"Route: {result['route_name']}")
        
        # Print summary statistics
        snapper.print_summary(results)
        
        # Save results to CSV
        snapper.save_results(results, output_csv_path)
        
        # Display sample results
        print(f"\nSample of results (first 5 clusters):")
        print("-" * 120)
        df_sample = pd.DataFrame(results[:5])
        print(df_sample[['cluster_number', 'original_lat', 'original_lon', 
                        'snapped_lat', 'snapped_lon', 'route_name', 
                        'snap_distance_meters']].to_string(index=False))
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please ensure the route JSON and cluster CSV files are in the correct location")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

# Additional utility functions for analysis

def analyze_snapping_quality(results_csv_path):
    """
    Analyze the quality of cluster snapping
    """
    df = pd.read_csv(results_csv_path)
    
    print("SNAPPING QUALITY ANALYSIS")
    print("=" * 50)
    
    # Distance statistics
    distances = df['snap_distance_meters']
    print(f"Distance Statistics:")
    print(f"  Mean: {distances.mean():.2f}m")
    print(f"  Std:  {distances.std():.2f}m")
    print(f"  Min:  {distances.min():.2f}m")
    print(f"  Max:  {distances.max():.2f}m")
    
    # Quality categories
    excellent = len(df[df['snap_distance_meters'] <= 100])
    good = len(df[(df['snap_distance_meters'] > 100) & (df['snap_distance_meters'] <= 500)])
    fair = len(df[(df['snap_distance_meters'] > 500) & (df['snap_distance_meters'] <= 1000)])
    poor = len(df[df['snap_distance_meters'] > 1000])
    
    print(f"\nQuality Distribution:")
    print(f"  Excellent (≤100m):     {excellent:2d} clusters ({excellent/len(df)*100:.1f}%)")
    print(f"  Good (100-500m):       {good:2d} clusters ({good/len(df)*100:.1f}%)")
    print(f"  Fair (500-1000m):      {fair:2d} clusters ({fair/len(df)*100:.1f}%)")
    print(f"  Poor (>1000m):         {poor:2d} clusters ({poor/len(df)*100:.1f}%)")

def export_for_visualization(results_csv_path, output_geojson_path):
    """
    Export results as GeoJSON for visualization
    """
    import json
    
    df = pd.read_csv(results_csv_path)
    
    features = []
    
    for _, row in df.iterrows():
        # Original cluster point
        orig_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row['original_lon'], row['original_lat']]
            },
            "properties": {
                "type": "original_cluster",
                "cluster_number": int(row['cluster_number']),
                "num_students": int(row['num_students'])
            }
        }
        
        # Snapped cluster point
        snapped_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row['snapped_lon'], row['snapped_lat']]
            },
            "properties": {
                "type": "snapped_cluster",
                "cluster_number": int(row['cluster_number']),
                "route_name": row['route_name'],
                "snap_distance": float(row['snap_distance_meters']),
                "num_students": int(row['num_students'])
            }
        }
        
        # Connection line
        connection_feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [row['original_lon'], row['original_lat']],
                    [row['snapped_lon'], row['snapped_lat']]
                ]
            },
            "properties": {
                "type": "snap_connection",
                "cluster_number": int(row['cluster_number']),
                "distance": float(row['snap_distance_meters'])
            }
        }
        
        features.extend([orig_feature, snapped_feature, connection_feature])
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_geojson_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"GeoJSON exported to {output_geojson_path}")