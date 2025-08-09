import pandas as pd
import numpy as np
import requests
import time

def create_distance_matrix_osrm(df, server_url='http://router.project-osrm.org'):
    """
    Create distance matrix using OSRM API and save as CSV
    """
    n_locations = len(df)
    distance_matrix = np.zeros((n_locations, n_locations))
    
    # OSRM batch processing
    batch_size = 50
    
    print(f"Processing {n_locations} locations in batches of {batch_size}...")
    
    # Process diagonal batches (same batch to same batch)
    total_batches = (n_locations + batch_size - 1) // batch_size
    processed_batches = 0
    
    for i in range(0, n_locations, batch_size):
        batch_end = min(i + batch_size, n_locations)
        batch_coords = []
        
        # Prepare coordinates string for OSRM
        for _, row in df.iloc[i:batch_end].iterrows():
            batch_coords.append(f"{row['longitude']},{row['latitude']}")
        
        coords_string = ";".join(batch_coords)
        
        try:
            # Make API request to OSRM
            url = f"{server_url}/table/v1/driving/{coords_string}"
            params = {
                'sources': ';'.join(map(str, range(len(batch_coords)))),
                'destinations': ';'.join(map(str, range(len(batch_coords))))
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                distances = np.array(data['distances']) if 'distances' in data else np.array(data['durations']) * 50
                
                # Fill the distance matrix
                for oi in range(len(batch_coords)):
                    for di in range(len(batch_coords)):
                        distance_matrix[i + oi][i + di] = distances[oi][di]
                
                processed_batches += 1
                print(f"Processed diagonal batch {processed_batches}/{total_batches}")
            else:
                print(f"Error with batch starting at {i}: {response.status_code}")
                
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
    
    # Fill the full matrix by making requests for cross-batch distances
    cross_batch_count = 0
    total_cross_batches = sum(1 for i in range(0, n_locations, batch_size) 
                             for j in range(i + batch_size, n_locations, batch_size))
    
    print(f"Processing {total_cross_batches} cross-batch combinations...")
    
    for i in range(0, n_locations, batch_size):
        for j in range(i + batch_size, n_locations, batch_size):
            # Calculate distances between different batches
            batch1_coords = []
            batch2_coords = []
            
            for _, row in df.iloc[i:min(i + batch_size, n_locations)].iterrows():
                batch1_coords.append(f"{row['longitude']},{row['latitude']}")
            
            for _, row in df.iloc[j:min(j + batch_size, n_locations)].iterrows():
                batch2_coords.append(f"{row['longitude']},{row['latitude']}")
            
            all_coords = batch1_coords + batch2_coords
            coords_string = ";".join(all_coords)
            
            try:
                url = f"{server_url}/table/v1/driving/{coords_string}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    distances = np.array(data['distances']) if 'distances' in data else np.array(data['durations']) * 50
                    
                    # Fill both triangular parts of the matrix
                    len1, len2 = len(batch1_coords), len(batch2_coords)
                    
                    for oi in range(len1):
                        for di in range(len2):
                            distance_matrix[i + oi][j + di] = distances[oi][len1 + di]
                            distance_matrix[j + di][i + oi] = distances[len1 + di][oi]
                
                cross_batch_count += 1
                print(f"Processed cross-batch {cross_batch_count}/{total_cross_batches}")
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing cross-batch {i}-{j}: {e}")
    
    return distance_matrix

def main():
    # Load your CSV file
    print("Loading data from snapped.csv...")
    df = pd.read_csv('../output/snapped.csv')

    print(f"Loaded {len(df)} locations")
    
    # Create distance matrix using OSRM
    distance_matrix = create_distance_matrix_osrm(df)
    
    # Create labels for rows and columns using cluster numbers
    labels = [f"cluster_{int(row['cluster_number'])}" for _, row in df.iterrows()]
    
    # Create DataFrame with proper labels
    distance_df = pd.DataFrame(distance_matrix, index=labels, columns=labels)
    
    # Save as CSV
    output_filename = '../output/distance_matrix.csv'
    distance_df.to_csv(output_filename)
    
    print(f"\n✓ Distance matrix saved as: {output_filename}")
    print(f"Matrix size: {distance_matrix.shape[0]} x {distance_matrix.shape[1]}")
    print(f"Total distances calculated: {distance_matrix.shape[0] * distance_matrix.shape[1]}")
    
    # Show sample of the matrix
    print(f"\nSample of distance matrix (first 5x5):")
    print(distance_df.iloc[:5, :5].round(2))
    
    # Show some statistics
    non_zero_distances = distance_matrix[distance_matrix > 0]
    if len(non_zero_distances) > 0:
        print(f"\nDistance Statistics:")
        print(f"Min distance: {np.min(non_zero_distances):.2f} meters")
        print(f"Max distance: {np.max(non_zero_distances):.2f} meters")
        print(f"Average distance: {np.mean(non_zero_distances):.2f} meters")

if __name__ == "__main__":
    main()