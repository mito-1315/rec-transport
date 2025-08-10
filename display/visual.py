import json
import folium
from itertools import cycle

# --- INPUT / OUTPUT ---
input_file = "routes_with_lat_lon.json"
output_html = "bus_stops_map.html"

# --- LOAD DATA ---
with open(input_file, "r") as f:
    routes_data = json.load(f)

# --- MAP CENTER ---
first_lat = routes_data["routes"][0]["stops"][0]["latitude"]
first_lon = routes_data["routes"][0]["stops"][0]["longitude"]
m = folium.Map(location=[first_lat, first_lon], zoom_start=12)

# --- COLORS FOR ROUTES ---
colors = cycle([
    "red", "blue", "green", "purple", "orange", "darkred", "lightblue",
    "darkgreen", "cadetblue", "pink", "gray", "black"
])

# --- ADD POINTS ---
for route_index, route in enumerate(routes_data["routes"], start=1):
    route_color = next(colors)
    for stop_index, stop in enumerate(route["stops"], start=1):
        lat = stop["latitude"]
        lon = stop["longitude"]
        demand = stop["demand"]

        label = f"Stop {stop_index} | Demand: {demand} | Route: {route_index}"
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=route_color,
            fill=True,
            fill_color=route_color,
            fill_opacity=0.9,
            popup=label,
            tooltip=label
        ).add_to(m)

# --- SAVE MAP ---
m.save(output_html)
print(f"Map saved to {output_html}")
