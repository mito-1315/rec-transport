import pandas as pd
import numpy as np
import requests
import time
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
output_filename = './output/distMatrixWithDepot.csv'
def haversine_distance(lat1, lon1, lat2, lon2):
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
    
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km * 1000  # Convert to meters

def get_osrm_route_distance(lat1, lon1, lat2, lon2, server_url='http://router.project-osrm.org'):
    """
    Get actual road distance between two points using OSRM route API
    """
    try:
        url = f"{server_url}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
        params = {
            'overview': 'false',
            'alternatives': 'false',
            'steps': 'false'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data['code'] == 'Ok' and len(data['routes']) > 0:
                # Return distance in meters
                return data['routes'][0]['distance']
            else:
                # Fallback to haversine distance if no route found
                return haversine_distance(lat1, lon1, lat2, lon2) * 1.3  # Add 30% for road routing
        else:
            # Fallback to haversine distance
            return haversine_distance(lat1, lon1, lat2, lon2) * 1.3
            
    except Exception as e:
        print(f"Error getting route distance: {e}")
        # Fallback to haversine distance
        return haversine_distance(lat1, lon1, lat2, lon2) * 1.3

def create_distance_matrix_proper_osrm(df, use_fallback=True):
    """
    Create distance matrix using OSRM route API for proper road distances
    """
    n_locations = len(df)
    distance_matrix = np.zeros((n_locations, n_locations))
    
    print(f"Creating distance matrix for {n_locations} locations...")
    print("This will take some time due to API rate limiting...")
    
    total_pairs = n_locations * (n_locations - 1) // 2  # Only calculate upper triangle
    
    # Progress bar for the matrix creation
    with tqdm(total=total_pairs, desc="Calculating distances", unit="pairs") as pbar:
        for i in range(n_locations):
            for j in range(i, n_locations):
                if i == j:
                    distance_matrix[i][j] = 0
                else:
                    lat1, lon1 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
                    lat2, lon2 = df.iloc[j]['latitude'], df.iloc[j]['longitude']
                    
                    # Get road distance using OSRM
                    if use_fallback:
                        # Use haversine with road factor for speed
                        distance = haversine_distance(lat1, lon1, lat2, lon2) * 1.3
                    else:
                        # Use actual OSRM routing (slower but more accurate)
                        distance = get_osrm_route_distance(lat1, lon1, lat2, lon2)
                        time.sleep(0.2)  # Rate limiting
                    
                    # Fill both sides of the matrix (symmetric)
                    distance_matrix[i][j] = distance
                    distance_matrix[j][i] = distance
                    
                    pbar.update(1)
    
    return distance_matrix

def create_distance_matrix_mapbox(df, api_key):
    """
    Alternative using Mapbox Matrix API (more reliable than OSRM)
    """
    n_locations = len(df)
    distance_matrix = np.zeros((n_locations, n_locations))
    
    # Mapbox allows up to 25x25 matrix per request
    batch_size = 25
    
    print(f"Creating distance matrix using Mapbox API...")
    
    total_batches = ((n_locations - 1) // batch_size + 1) ** 2
    
    with tqdm(total=total_batches, desc="Processing batches", unit="batch") as pbar:
        for i in range(0, n_locations, batch_size):
            for j in range(0, n_locations, batch_size):
                batch_end_i = min(i + batch_size, n_locations)
                batch_end_j = min(j + batch_size, n_locations)
                
                # Prepare coordinates
                origins = []
                destinations = []
                
                for oi in range(i, batch_end_i):
                    row = df.iloc[oi]
                    origins.append(f"{row['longitude']},{row['latitude']}")
                
                for di in range(j, batch_end_j):
                    row = df.iloc[di]
                    destinations.append(f"{row['longitude']},{row['latitude']}")
                
                # Create coordinate string
                all_coords = list(set(origins + destinations))  # Remove duplicates
                coords_string = ";".join(all_coords)
                
                try:
                    url = f"https://api.mapbox.com/directions-matrix/v1/mapbox/driving/{coords_string}"
                    params = {
                        'access_token': api_key,
                        'sources': ';'.join([str(all_coords.index(coord)) for coord in origins]),
                        'destinations': ';'.join([str(all_coords.index(coord)) for coord in destinations])
                    }
                    
                    response = requests.get(url, params=params)
                    
                    if response.status_code == 200:
                        data = response.json()
                        distances = data['distances']
                        
                        # Fill the matrix
                        for oi, origin_distances in enumerate(distances):
                            for di, distance in enumerate(origin_distances):
                                if distance is not None:
                                    distance_matrix[i + oi][j + di] = distance
                                else:
                                    # Fallback to haversine
                                    lat1, lon1 = df.iloc[i + oi]['latitude'], df.iloc[i + oi]['longitude']
                                    lat2, lon2 = df.iloc[j + di]['latitude'], df.iloc[j + di]['longitude']
                                    distance_matrix[i + oi][j + di] = haversine_distance(lat1, lon1, lat2, lon2) * 1.3
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    tqdm.write(f"Error processing batch {i}-{j}: {e}")
                    # Fill with haversine distances as fallback
                    for oi in range(len(origins)):
                        for di in range(len(destinations)):
                            lat1, lon1 = df.iloc[i + oi]['latitude'], df.iloc[i + oi]['longitude']
                            lat2, lon2 = df.iloc[j + di]['latitude'], df.iloc[j + di]['longitude']
                            distance_matrix[i + oi][j + di] = haversine_distance(lat1, lon1, lat2, lon2) * 1.3
                
                pbar.update(1)
    
    return distance_matrix

def main():
    # Load your CSV file
    print("Loading data from merged_mdvrp_data.csv...")
    df = pd.read_csv('merged_mdvrp_data.csv')
    
    print(f"Loaded {len(df)} locations")
    
    # Choose method:
    # Option 1: Fast method using haversine distance with road factor (recommended for large datasets)
    print("\nChoose distance calculation method:")
    print("1. Fast method (Haversine + road factor) - Recommended")
    print("2. OSRM route API (slow but accurate)")
    print("3. Mapbox API (requires API key)")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        print("Using fast method with road distance approximation...")
        distance_matrix = create_distance_matrix_proper_osrm(df, use_fallback=True)
    elif choice == "2":
        print("Using OSRM route API (this will take a long time)...")
        distance_matrix = create_distance_matrix_proper_osrm(df, use_fallback=False)
    elif choice == "3":
        api_key = input("Enter your Mapbox API key: ").strip()
        distance_matrix = create_distance_matrix_mapbox(df, api_key)
    else:
        print("Invalid choice, using fast method...")
        distance_matrix = create_distance_matrix_proper_osrm(df, use_fallback=True)
    
    # Create labels for rows and columns using stop_id
    labels = [f"{row['stop_id']}" for _, row in df.iterrows()]
    
    # Create DataFrame with proper labels
    distance_df = pd.DataFrame(distance_matrix, index=labels, columns=labels)
    
    # Save as CSV
    
    distance_df.to_csv(output_filename)
    
    print(f"\n✓ Distance matrix saved as: {output_filename}")
    print(f"Matrix size: {distance_matrix.shape[0]} x {distance_matrix.shape[1]}")
    print(f"Total distances calculated: {distance_matrix.shape[0] * distance_matrix.shape[1]}")
    
    # Show sample of the matrix
    print(f"\nSample of distance matrix (first 5x5):")
    print(distance_df.iloc[:5, :5].round(0))
    
    # Show some statistics
    non_zero_distances = distance_matrix[distance_matrix > 0]
    if len(non_zero_distances) > 0:
        print(f"\nDistance Statistics:")
        print(f"Min distance: {np.min(non_zero_distances):.0f} meters")
        print(f"Max distance: {np.max(non_zero_distances):.0f} meters")
        print(f"Average distance: {np.mean(non_zero_distances):.0f} meters")
        print(f"Median distance: {np.median(non_zero_distances):.0f} meters")

if __name__ == "__main__":
    main()