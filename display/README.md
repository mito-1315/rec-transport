# CSV Coordinate Viewer

A web-based tool to upload CSV files containing latitude/longitude coordinates and visualize them on an interactive map.

## Features

- 📁 Upload CSV files with coordinate data
- 🗺️ Interactive map visualization using Folium
- 📍 Markers with popup information
- 🚌 Support for bus route data with connecting lines
- 📊 Statistics display (total points, routes, named stops)
- 🎨 Color-coded markers by route
- ✅ Data validation and error handling

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python csv_coordinate_viewer.py
```

3. Open your browser and go to: http://localhost:5000

## CSV File Format

### Required Columns
- `latitude` - Decimal degrees (-90 to 90)
- `longitude` - Decimal degrees (-180 to 180)

### Optional Columns
- `stop_name` - Name of the bus stop/location
- `route_number` - Bus route identifier
- `stop_number` - Stop sequence number
- `sequence` - Order of stops in route

### Example CSV:
```csv
latitude,longitude,stop_name,route_number,stop_number,sequence
13.0827,80.2707,Marina Beach,102A,1,1
13.0878,80.2785,Lighthouse,102A,2,2
13.0825,80.2750,Anna Square,102A,3,3
```

## Usage

1. **Upload CSV**: Click "Choose File" and select your CSV file
2. **View Map**: The map will automatically display with markers for each coordinate
3. **Explore**: Click on markers to see detailed information
4. **Routes**: If route data is available, routes will be connected with colored lines

## Features

### Map Markers
- Different colors for different routes
- Popup information with coordinates and stop details
- Tooltips showing stop names
- Bus icon markers

### Route Visualization
- Connected lines between stops in sequence
- Color-coded by route number
- Automatic route grouping

### Data Validation
- Checks for required columns
- Validates coordinate ranges
- Removes invalid data points
- Shows error messages for issues

## Troubleshooting

### Common Issues

1. **"Missing required columns" error**
   - Ensure your CSV has `latitude` and `longitude` columns
   - Check column name spelling (case-sensitive)

2. **"No valid coordinate data found" error**
   - Check that coordinates are in decimal degrees format
   - Ensure no empty cells in coordinate columns

3. **Map not displaying properly**
   - Check that coordinates are within valid ranges
   - Ensure you have internet connection (for map tiles)

### File Requirements
- Maximum file size: 16MB
- File format: CSV only
- Headers must be in first row
- Coordinates in decimal degrees format

## Example Files

You can test the viewer with these sample coordinates:

**Chennai Bus Stops:**
```csv
latitude,longitude,stop_name,route_number
13.0827,80.2707,Marina Beach,102A
13.0878,80.2785,Lighthouse,102A
13.0825,80.2750,Anna Square,101B
13.0845,80.2720,Central Station,101B
```

Save this as a `.csv` file and upload it to test the functionality.
