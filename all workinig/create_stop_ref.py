# Create this script: rec-transport/all workinig/create_bus_stops.py
import pandas as pd
import numpy as np

# Load the MTC stops data
mtc_stops = pd.read_csv('../output/mtc_all_stops_20250805_184547.csv')

# Load geocoded data if available
try:
    geocoded_stops = pd.read_csv('../output/bus_stops_with_coordinates.csv')
    # Merge with MTC data
    bus_stops = pd.merge(mtc_stops, geocoded_stops, on='stop_name', how='left')
except:
    # If no geocoded data, create dummy coordinates (you'll need to geocode later)
    bus_stops = mtc_stops.copy()
    bus_stops['latitude'] = np.nan
    bus_stops['longitude'] = np.nan

# Create required format
bus_stops['cluster_id'] = range(1, len(bus_stops) + 1)
bus_stops['student_count'] = 50  # Default capacity

# Select required columns
final_bus_stops = bus_stops[['latitude', 'longitude', 'cluster_id', 'student_count', 'stop_name']]

# Save
final_bus_stops.to_csv('bus_stops_ref.csv', index=False)