import json
import folium
import openrouteservice
import time

# Config
ORS_API_KEY = 'eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImNlMTRlODRmYzE0NzRlZjZiZWRmODJiYmNjMDY5NTNlIiwiaCI6Im11cm11cjY0In0='
REQUEST_DELAY = 1.1  # seconds between API calls

# Load the JSON produced from the previous step
with open('../output/mock-mtc-coords.json', 'r') as f:
    data = json.load(f)

routes = data["routes"]
unique_stops = data["unique_stops"]

# Set up map center (e.g., average of all stops)
all_coords = [(stop['latitude'], stop['longitude']) for stop in unique_stops]
midpoint = (sum(lat for lat, lon in all_coords) / len(all_coords),
            sum(lon for lat, lon in all_coords) / len(all_coords))
m = folium.Map(location=midpoint, zoom_start=11, tiles='cartodbpositron')

# ORS client
client = openrouteservice.Client(key=ORS_API_KEY)

# Deduplicate stops (for global marker plotting)
added_stop_keys = set()

for route_num, stops in routes.items():
    # Only plot if route has at least 2 consecutive stops
    if len(stops) < 2:
        continue

    prev_stop = None
    for stop in stops:
        key = (stop['latitude'], stop['longitude'])
        print(key)
        # Add one marker per unique stop
        if key not in added_stop_keys:
            folium.CircleMarker(
                location=[stop['latitude'], stop['longitude']],
                radius=4, color='darkblue', fill=True, fill_opacity=0.65,
                popup=folium.Popup(html=stop["stop_name"])
            ).add_to(m)
            added_stop_keys.add(key)
        
        if prev_stop:
            # Call ORS for precise route between prev_stop and current stop
            coords = [
                (prev_stop['longitude'], prev_stop['latitude']),
                (stop['longitude'], stop['latitude'])
            ]
            try:
                result = client.directions(coords, profile='driving-car', format='geojson')
                route_line = [
                    [pt[1], pt[0]] for pt in result['features'][0]['geometry']['coordinates']
                ]
                folium.PolyLine(
                    route_line,
                    color='blue', weight=3, opacity=0.7
                ).add_to(m)
            except Exception as e:
                print(f"Failed for {prev_stop['stop_name']} -> {stop['stop_name']}: {e}")
            time.sleep(REQUEST_DELAY)  # Pause to respect API rate limits
        prev_stop = stop

# Save or display
m.save('bus_routes_map.html')
