import pandas as pd
import numpy as np
import googlemaps
import time
import json
import os

# Read your CSV file
df = pd.read_csv('../output/snapped.csv')

# Initialize Google Maps client
gmaps = googlemaps.Client(key='AIzaSyA_KVzNylr1qdcLSIPIBkhR77w09crzUkQ')

def create_distance_matrix_google(df, mode='driving'):
    """
    Create distance matrix using Google Maps Distance Matrix API
    """
    n_locations = len(df)
    distance_matrix = np.zeros((n_locations, n_locations))
    
    # Google API allows max 25 origins and 25 destinations per request
    batch_size = 25
    total_batches = ((n_locations - 1) // batch_size + 1) ** 2
    current_batch = 0
    
    print(f"Processing {n_locations} locations in {total_batches} batches...")
    
    for i in range(0, n_locations, batch_size):
        for j in range(0, n_locations, batch_size):
            current_batch += 1
            print(f"Processing batch {current_batch}/{total_batches}")
            
            # Get batch of origins and destinations
            origins = [(row['latitude'], row['longitude']) 
                      for _, row in df.iloc[i:i+batch_size].iterrows()]
            destinations = [(row['latitude'], row['longitude']) 
                           for _, row in df.iloc[j:j+batch_size].iterrows()]
            
            try:
                # Make API request
                result = gmaps.distance_matrix(
                    origins=origins,
                    destinations=destinations,
                    mode=mode,
                    units='metric'
                )
                
                # Parse results
                for oi, origin_result in enumerate(result['rows']):
                    for di, dest_result in enumerate(origin_result['elements']):
                        if dest_result['status'] == 'OK':
                            distance = dest_result['distance']['value']  # in meters
                            distance_matrix[i + oi][j + di] = distance
                        else:
                            distance_matrix[i + oi][j + di] = np.inf
                
                # Rate limiting - Google allows 1000 requests per day for free tier
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing batch {i}-{j}: {e}")
    
    return distance_matrix

# Create output directory if it doesn't exist
output_dir = '../output'
os.makedirs(output_dir, exist_ok=True)

# Execute the distance matrix creation
print("Creating distance matrix using Google Maps API...")
distance_matrix = create_distance_matrix_google(df)

# Create DataFrame with proper indices
location_indices = [f"location_{i}" for i in range(len(df))]
distance_df = pd.DataFrame(
    distance_matrix, 
    index=location_indices, 
    columns=location_indices
)

# Save as CSV
csv_filename = os.path.join(output_dir, 'distance_matrix_google.csv')
distance_df.to_csv(csv_filename)
print(f"Distance matrix saved as CSV: {csv_filename}")

# Save as JSON
json_filename = os.path.join(output_dir, 'distance_matrix_google.json')
distance_dict = distance_df.to_dict()
with open(json_filename, 'w') as f:
    json.dump(distance_dict, f, indent=2)
print(f"Distance matrix saved as JSON: {json_filename}")

# Print summary statistics
print(f"\nDistance Matrix Summary:")
print(f"Shape: {distance_matrix.shape}")

# Check for valid distances
valid_distances = distance_matrix[(distance_matrix > 0) & (distance_matrix != np.inf)]
if len(valid_distances) > 0:
    print(f"Min distance: {np.min(valid_distances):.2f} meters")
    print(f"Max distance: {np.max(valid_distances):.2f} meters")
    print(f"Mean distance: {np.mean(valid_distances):.2f} meters")
    print(f"Number of valid distances: {len(valid_distances)}")
else:
    print("No valid distances found in the matrix")
    
# Print counts of different types of values
zero_count = np.sum(distance_matrix == 0)
inf_count = np.sum(distance_matrix == np.inf)
valid_count = len(valid_distances)
total_count = distance_matrix.size

print(f"Zero distances: {zero_count}")
print(f"Infinite distances: {inf_count}")
print(f"Valid distances: {valid_count}")
print(f"Total entries: {total_count}")