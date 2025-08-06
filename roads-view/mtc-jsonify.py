import pandas as pd
import json

# Read the input CSV
df = pd.read_csv('../output/mtc_optimized_coordinates.csv')

# Drop rows without valid coordinates (using snapped_latitude/longitude for precision)
df_valid = df.dropna(subset=['snapped_latitude', 'snapped_longitude'])

# Build unique stop dictionary (to deduplicate globally)
stop_key = lambda r: (round(r['snapped_latitude'], 6), round(r['snapped_longitude'], 6))
unique_stops = {}
for _, row in df_valid.iterrows():
    key = stop_key(row)
    if key not in unique_stops:
        unique_stops[key] = {
            'stop_name': row['stop_name'],
            'latitude': row['snapped_latitude'],
            'longitude': row['snapped_longitude']
        }

# Build routes as ordered lists of stops (using the deduped stops' coordinates)
routes = {}
for route_number, group in df_valid.groupby('route_number'):
    stops = []
    group_sorted = group.sort_values(by='sequence')
    for _, row in group_sorted.iterrows():
        key = stop_key(row)
        stops.append({
            'stop_name': row['stop_name'],
            'latitude': unique_stops[key]['latitude'],
            'longitude': unique_stops[key]['longitude'],
            'sequence': row['sequence']
        })
    routes[str(route_number)] = stops

# Output JSON structure
output = {
    "routes": routes,
    "unique_stops": [
        {
            "stop_name": stop['stop_name'],
            "latitude": stop['latitude'],
            "longitude": stop['longitude']
        }
        for stop in unique_stops.values()
    ]
}

# Save to file, or just print
#print(json.dumps(output, indent=2))
with open('../output/mtc-coords.json', 'w') as f:
    json.dump(output, f, indent=2)
