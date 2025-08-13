import pandas as pd
import folium
from folium.plugins import MarkerCluster
import requests
import json
import time
import os
from datetime import datetime

# Get current timestamp for output filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define file paths
stop_sequence_path = 'route_optimization_output/stop_sequences_20250813_112226.csv'
route_summary_path = 'route_optimization_output/route_summary_20250813_112226.csv'
depot_assignments_path = 'route_optimization_output/depot_assignments_20250813_112226.csv'

# Read CSV files
stop_sequence_df = pd.read_csv(stop_sequence_path)
route_summary_df = pd.read_csv(route_summary_path)
depot_assignments_df = pd.read_csv(depot_assignments_path)

# Print column names to debug
print("Stop Sequence columns:", stop_sequence_df.columns.tolist())
print("Route Summary columns:", route_summary_df.columns.tolist())
print("Depot Assignments columns:", depot_assignments_df.columns.tolist())

# Define college coordinates
college_coords = (13.008794724595475, 80.00342657961114)

# Create map
m = folium.Map(location=college_coords, zoom_start=11, tiles="OpenStreetMap")

# Add college marker
folium.Marker(
    location=college_coords,
    popup="College",
    tooltip="College",
    icon=folium.Icon(color="red", icon="university", prefix='fa')
).add_to(m)

# Define a color palette for different bus routes
colors = [
    '#0000FF', '#008000', '#800080', '#FFA500', '#8B0000',  # blue, green, purple, orange, dark red
    '#FF6347', '#4B0082', '#006400', '#FF4500', '#00008B',  # tomato, indigo, dark green, orange red, dark blue
    '#8A2BE2', '#228B22', '#DC143C', '#00FFFF', '#FF00FF',  # blue violet, forest green, crimson, cyan, magenta
    '#FFD700', '#20B2AA', '#9932CC', '#FF1493', '#00FA9A'   # gold, light sea green, dark orchid, deep pink, medium spring green
]

# Add depot markers
for index, depot in depot_assignments_df.iterrows():
    folium.Marker(
        location=[depot['Depot_Latitude'], depot['Depot_Longitude']],
        popup=folium.Popup(f"""
            <b>Depot:</b> {depot['Depot_Name']}<br>
            <b>Assigned Students:</b> {depot['Assigned_Students']}<br>
            <b>Number of Routes:</b> {depot['Number_of_Routes_Assigned']}<br>
            <b>Bus IDs:</b> {depot['Assigned_Bus_IDs']}
        """, max_width=300),
        tooltip=f"Depot: {depot['Depot_Name']}",
        icon=folium.Icon(color="darkgreen", icon="warehouse", prefix='fa')
    ).add_to(m)

# Create a feature group for each bus
bus_groups = {}

# OpenRouteService API key
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImU5MDczZGYyYzE2ODRjN2I4MDdlMjQ5M2NjODllNjMwIiwiaCI6Im11cm11cjY0In0="  # Replace this with your actual API key

# Function to get route using OpenRouteService
def get_ors_route(start_coords, end_coords, retry=3, backoff=2):
    """Get route between two points with retry logic"""
    coords = [
        [start_coords[1], start_coords[0]],  # OpenRouteService uses [lng, lat] format
        [end_coords[1], end_coords[0]]
    ]
    
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': ORS_API_KEY,
        'Content-Type': 'application/json; charset=utf-8'
    }
    
    data = {
        "coordinates": coords,
        "format": "geojson"
    }
    
    for attempt in range(retry):
        try:
            response = requests.post(
                'https://api.openrouteservice.org/v2/directions/driving-car',
                json=data,
                headers=headers
            )
            
            if response.status_code == 200:
                route_data = response.json()
                # Debug output to see response structure
                print(f"Response keys: {route_data.keys()}")
                return route_data
            elif response.status_code == 429:  # Rate limit exceeded
                wait_time = backoff * (2 ** attempt)
                print(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                print(f"Error: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            print(f"Error getting route on attempt {attempt+1}: {e}")
            if attempt < retry - 1:  # i.e. not the last attempt
                wait_time = backoff * (2 ** attempt)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return None
    
    return None

# For debugging: Draw a straight line fallback function
def draw_straight_line(start, end, color, bus_id, label, feature_group):
    folium.PolyLine(
        [start, end],
        color=color,
        weight=4,
        opacity=0.7,
        tooltip=f"Straight line for {bus_id}: {label}"
    ).add_to(feature_group)

# Status tracking
total_buses = len(route_summary_df)
processed_buses = 0

# Limit processing to first few buses for testing
# max_buses = 2  # Uncomment to limit the number of buses processed
print(f"Processing bus routes...")

# Process stops and routes
for index, bus in route_summary_df.iterrows():
    # Uncomment to limit processing
    # if index >= max_buses:
    #     break
        
    bus_id = bus['Bus_ID']
    color = colors[index % len(colors)]
    
    print(f"Processing bus {bus_id} ({processed_buses+1}/{total_buses})...")
    
    bus_groups[bus_id] = folium.FeatureGroup(name=f"Bus {bus_id}")
    
    # Get bus stops
    bus_stops = stop_sequence_df[stop_sequence_df['Bus_ID'] == bus_id].sort_values(by='Sequence_Number')
    
    # If no stops found, skip this bus
    if len(bus_stops) == 0:
        print(f"  No stops found for bus {bus_id}, skipping...")
        processed_buses += 1
        continue
    
    # Get depot information for this bus
    depot_info = None
    for _, depot in depot_assignments_df.iterrows():
        assigned_buses = str(depot['Assigned_Bus_IDs']).split(',')
        assigned_buses = [b.strip() for b in assigned_buses]
        if bus_id in assigned_buses:
            depot_info = depot
            break
    
    if depot_info is not None:
        depot_coords = (depot_info['Depot_Latitude'], depot_info['Depot_Longitude'])
    else:
        # Use route end coordinates if depot not found
        depot_coords = (bus['End_Latitude'], bus['End_Longitude'])
    
    # Add bus stops to the map
    stops_coords = []
    for _, stop in bus_stops.iterrows():
        if pd.notna(stop['Latitude']) and pd.notna(stop['Longitude']):
            stop_coords = (stop['Latitude'], stop['Longitude'])
            stops_coords.append(stop_coords)
            
            # Create stop marker
            folium.CircleMarker(
                location=stop_coords,
                radius=5 + min(stop['Students_Count'] / 2, 10),  # Size based on student count
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=folium.Popup(f"""
                    <b>Stop:</b> {stop['Stop_Name']}<br>
                    <b>Stop ID:</b> {stop['Stop_ID']}<br>
                    <b>Sequence:</b> {stop['Sequence_Number']}<br>
                    <b>Students:</b> {stop['Students_Count']}<br>
                    <b>Bus:</b> {bus_id}
                """, max_width=300),
                tooltip=f"Stop {stop['Sequence_Number']}: {stop['Stop_Name']} ({stop['Students_Count']} students)"
            ).add_to(bus_groups[bus_id])
    
    # Add bus info marker at college
    folium.Marker(
        location=college_coords,
        popup=folium.Popup(f"""
            <b>Bus:</b> {bus_id}<br>
            <b>Total Students:</b> {bus['Total_Students']}<br>
            <b>Capacity Utilization:</b> {bus['Capacity_Utilization_%']}%<br>
            <b>Number of Stops:</b> {bus['Number_of_Stops']}<br>
            <b>Estimated Distance:</b> {bus['Estimated_Distance_KM']} km<br>
            <b>Estimated Time:</b> {bus['Estimated_Time_Minutes']} mins
        """, max_width=300),
        tooltip=f"Bus {bus_id} Info",
        icon=folium.DivIcon(
            icon_size=(20, 20),
            icon_anchor=(10, 10),
            html=f'<div style="background-color:{color};width:20px;height:20px;border-radius:50%;text-align:center;line-height:20px;color:white;font-weight:bold;">{index+1}</div>',
        )
    ).add_to(bus_groups[bus_id])
    
    # Add routes between stops
    if stops_coords:
        # First route: College to first stop
        first_stop = stops_coords[0]
        print(f"  Getting route: College to first stop...")
        
        # Try to get the route data from OpenRouteService
        route_json = get_ors_route(college_coords, first_stop)
        
        # Check if route_json is valid and has the right structure
        if route_json and 'features' in route_json and len(route_json['features']) > 0:
            try:
                route_coords = [[point[1], point[0]] for point in route_json["features"][0]["geometry"]["coordinates"]]
                folium.PolyLine(
                    route_coords,
                    color=color,
                    weight=4,
                    opacity=0.7,
                    tooltip=f"Route for {bus_id}: College to first stop"
                ).add_to(bus_groups[bus_id])
            except (KeyError, IndexError) as e:
                print(f"  Error processing route data: {e}")
                # Fallback to straight line
                print(f"  Using straight line fallback for College to first stop")
                draw_straight_line(college_coords, first_stop, color, bus_id, "College to first stop", bus_groups[bus_id])
        else:
            # Fallback to straight line if API fails
            print(f"  Using straight line fallback for College to first stop")
            draw_straight_line(college_coords, first_stop, color, bus_id, "College to first stop", bus_groups[bus_id])
        
        # For each stop, draw route to next stop
        for i in range(len(stops_coords) - 1):
            start = stops_coords[i]
            end = stops_coords[i + 1]
            print(f"  Getting route: Stop {i+1} to Stop {i+2}...")
            
            # Get route between these stops
            route_json = get_ors_route(start, end)
            
            # Check if route_json is valid and has the right structure
            if route_json and 'features' in route_json and len(route_json['features']) > 0:
                try:
                    route_coords = [[point[1], point[0]] for point in route_json["features"][0]["geometry"]["coordinates"]]
                    folium.PolyLine(
                        route_coords,
                        color=color,
                        weight=4,
                        opacity=0.7,
                        tooltip=f"Route for {bus_id}: Stop {i+1} to Stop {i+2}"
                    ).add_to(bus_groups[bus_id])
                except (KeyError, IndexError) as e:
                    print(f"  Error processing route data: {e}")
                    # Fallback to straight line
                    print(f"  Using straight line fallback for Stop {i+1} to Stop {i+2}")
                    draw_straight_line(start, end, color, bus_id, f"Stop {i+1} to Stop {i+2}", bus_groups[bus_id])
            else:
                # Fallback to straight line if API fails
                print(f"  Using straight line fallback for Stop {i+1} to Stop {i+2}")
                draw_straight_line(start, end, color, bus_id, f"Stop {i+1} to Stop {i+2}", bus_groups[bus_id])
            
            # Add some delay to avoid hitting API rate limits
            time.sleep(0.5)
        
        # Last route: Last stop to depot
        last_stop = stops_coords[-1]
        print(f"  Getting route: Last stop to depot...")
        
        # Get route from last stop to depot
        route_json = get_ors_route(last_stop, depot_coords)
        
        # Check if route_json is valid and has the right structure
        if route_json and 'features' in route_json and len(route_json['features']) > 0:
            try:
                route_coords = [[point[1], point[0]] for point in route_json["features"][0]["geometry"]["coordinates"]]
                folium.PolyLine(
                    route_coords,
                    color=color,
                    weight=4,
                    opacity=0.7,
                    tooltip=f"Route for {bus_id}: Last stop to depot"
                ).add_to(bus_groups[bus_id])
            except (KeyError, IndexError) as e:
                print(f"  Error processing route data: {e}")
                # Fallback to straight line
                print(f"  Using straight line fallback for Last stop to depot")
                draw_straight_line(last_stop, depot_coords, color, bus_id, "Last stop to depot", bus_groups[bus_id])
        else:
            # Fallback to straight line if API fails
            print(f"  Using straight line fallback for Last stop to depot")
            draw_straight_line(last_stop, depot_coords, color, bus_id, "Last stop to depot", bus_groups[bus_id])
    
    # Add the feature group to the map
    bus_groups[bus_id].add_to(m)
    
    processed_buses += 1
    print(f"Completed bus {bus_id} ({processed_buses}/{total_buses})")

# Add layer control to toggle bus routes
folium.LayerControl().add_to(m)

# Add mini map for context
from folium.plugins import MiniMap
minimap = MiniMap()
m.add_child(minimap)

# Add fullscreen control
from folium.plugins import Fullscreen
Fullscreen().add_to(m)

# Add measure tool
from folium.plugins import MeasureControl
m.add_child(MeasureControl())

# Create a legend
legend_html = '''
<div style="position: fixed; 
     bottom: 50px; left: 50px; width: 220px; height: auto;
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color: white; padding: 10px;
     overflow-y: auto; max-height: 300px;">
     <b>Bus Routes Legend</b><br>
'''

for index, bus in route_summary_df.iterrows():
    color = colors[index % len(colors)]
    legend_html += f'<i class="fa fa-circle" style="color:{color}"></i> Bus {bus["Bus_ID"]} - {bus["Total_Students"]} students<br>'

legend_html += '''
     <br><b>Markers:</b><br>
     <i class="fa fa-university" style="color:red"></i> College<br>
     <i class="fa fa-warehouse" style="color:darkgreen"></i> Depot<br>
     <i class="fa fa-circle"></i> Bus Stop (size = student count)<br>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Save map
output_filename = f'bus_routes_map_{timestamp}.html'
m.save(output_filename)
print(f"Map saved as {output_filename}")
print("NOTE: Make sure to set your OpenRouteService API key before running!")