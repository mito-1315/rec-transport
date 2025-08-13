import pandas as pd
import numpy as np
from geopy.distance import geodesic
import datetime
import os

# Get the correct path to the merged data file
# Since this script runs from the data/ directory, we need to go up one level
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
merged_file_path = os.path.join(parent_dir, 'Output_data', 'merged_student_data_20250811_154529.csv')

# Check if the merged file exists
if not os.path.exists(merged_file_path):
    print(f"Error: Merged data file not found at: {merged_file_path}")
    print("Please run the merge script first to create the merged dataset.")
    exit(1)

print(f"Loading merged data from: {merged_file_path}")

# Load data
student_locations = pd.read_csv(merged_file_path)
print(f"Loaded {len(student_locations)} student records")

# Chennai city center coordinates (approximate)
#13.009913, 80.002021
chennai_center = (13.009913, 80.002021)
max_reasonable_distance_km = 65  # Maximum reasonable distance from city center

# Function to filter out unreasonable points
def is_valid_location(row):
    # Check if within reasonable distance from Chennai
    distance = geodesic(chennai_center, (row['latitude'], row['longitude'])).km
    return distance <= max_reasonable_distance_km

print("Filtering locations...")
# Apply filter
valid_locations = student_locations[student_locations.apply(is_valid_location, axis=1)]
outlier_locations = student_locations[~student_locations.apply(is_valid_location, axis=1)]

print(f"Removed {len(student_locations) - len(valid_locations)} invalid data points")

# Create output directory if it doesn't exist
output_dir = os.path.join(script_dir, 'Output Data')
os.makedirs(output_dir, exist_ok=True)

# Generate filename with current date and time
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"cleaned_student_locations_{current_time}.csv"
outliers_filename = f"outliers_student_locations_{current_time}.csv"
output_path = os.path.join(output_dir, output_filename)
outliers_path = os.path.join(output_dir, outliers_filename)

# Save cleaned data to new file
valid_locations.to_csv(output_path, index=False)
outlier_locations.to_csv(outliers_path, index=False)

print(f"Cleaned data saved to: {output_path}")
print(f"Outliers data saved to: {outliers_path}")
print(f"Total valid locations: {len(valid_locations)}")
print(f"Total outlier locations: {len(outlier_locations)}")
print(f"Data cleaning completed successfully!")