import pandas as pd
import json

# Read the CSV file
depot_stop_df = pd.read_csv('depot-stop-data.csv')

# Read the optimized routes JSON
with open('optimized_routes.json', 'r') as f:
    optimized_routes = json.load(f)

# Create a dictionary for quick lookup of coordinates
coord_lookup = {}
for _, row in depot_stop_df.iterrows():
    coord_lookup[row['stop_id']] = {
        'latitude': row['latitude'],
        'longitude': row['longitude']
    }

# Process each route and add coordinates
routes_with_coords = []

for route in optimized_routes['routes']:
    route_with_coords = {
        'vehicle_id': route['vehicle_id'],
        'stops': []
    }
    
    for stop in route['stops']:
        stop_id = stop['stop_id']
        stop_with_coords = {
            'stop_id': stop_id,
            'demand': stop['demand'],
            'cumulative_load': stop['cumulative_load']
        }
        
        # Add coordinates if available
        if stop_id in coord_lookup:
            stop_with_coords['latitude'] = coord_lookup[stop_id]['latitude']
            stop_with_coords['longitude'] = coord_lookup[stop_id]['longitude']
        
        route_with_coords['stops'].append(stop_with_coords)
    
    routes_with_coords.append(route_with_coords)

# Create the final output structure
output = {
    'routes': routes_with_coords
}

# Save to JSON file
with open('routes_with_lat_lon_merged.json', 'w') as f:
    json.dump(output, f, indent=2)

print("Successfully merged depot-stop-data.csv and optimized_routes.json")
print(f"Created routes_with_lat_lon_merged.json with {len(routes_with_coords)} routes")