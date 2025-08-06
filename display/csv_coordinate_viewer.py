#!/usr/bin/env python3
"""
CSV Coordinate Viewer
A web application to upload CSV files and display latitude/longitude coordinates
on an interactive map along with stop names.
"""

import os
import pandas as pd
import folium
from flask import Flask, request, render_template_string, redirect, url_for, flash, send_file
import tempfile
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)
app.secret_key = 'csv-coordinate-viewer-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# HTML template for the main page
MAIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>CSV Coordinate Viewer</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .upload-section { margin-bottom: 30px; padding: 20px; border: 2px dashed #ddd; border-radius: 10px; }
        .file-input { margin: 10px 0; }
        .btn { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 14px; }
        .btn:hover { background-color: #0056b3; }
        .info { background-color: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .error { background-color: #ffe7e7; padding: 15px; border-radius: 5px; margin: 20px 0; color: #d00; }
        .success { background-color: #e7ffe7; padding: 15px; border-radius: 5px; margin: 20px 0; color: #080; }
        .stats { display: flex; gap: 20px; margin: 20px 0; }
        .stat-card { flex: 1; background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }
        .stat-number { font-size: 24px; font-weight: bold; color: #007bff; }
        .map-container { margin-top: 20px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 12px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .sample-table { max-width: 600px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🗺️ CSV Coordinate Viewer</h1>
        
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="{{ category }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <div class="info">
            <h3>📋 Instructions:</h3>
            <ul>
                <li><strong>Required columns:</strong> <code>latitude</code>, <code>longitude</code></li>
                <li><strong>Optional columns:</strong> <code>stop_name</code>, <code>route_number</code>, <code>stop_number</code>, <code>sequence</code></li>
                <li>CSV file should have headers in the first row</li>
                <li>Coordinates should be in decimal degrees format</li>
            </ul>
        </div>
        
        <div class="upload-section">
            <h3>📁 Upload CSV File</h3>
            <form method="POST" enctype="multipart/form-data">
                <div class="file-input">
                    <input type="file" name="csv_file" accept=".csv" required>
                </div>
                <button type="submit" class="btn">🚀 Upload and Visualize</button>
            </form>
        </div>
        
        {% if data %}
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{{ data|length }}</div>
                    <div>Total Points</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{{ routes_count }}</div>
                    <div>Unique Routes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{{ stops_with_names }}</div>
                    <div>Named Stops</div>
                </div>
            </div>
            
            <div class="map-container">
                {{ map_html|safe }}
            </div>
        {% endif %}
        
        <div class="info sample-table">
            <h3>📊 Sample CSV Format:</h3>
            <table>
                <tr><th>latitude</th><th>longitude</th><th>stop_name</th><th>route_number</th></tr>
                <tr><td>13.0827</td><td>80.2707</td><td>Marina Beach</td><td>102A</td></tr>
                <tr><td>13.0878</td><td>80.2785</td><td>Lighthouse</td><td>102A</td></tr>
            </table>
        </div>
    </div>
</body>
</html>
"""

def create_map(data):
    """Create a Folium map with the coordinate data"""
    if data.empty:
        return None
    
    # Calculate map center
    center_lat = data['latitude'].mean()
    center_lon = data['longitude'].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add markers for each point
    for idx, row in data.iterrows():
        # Create popup text
        popup_text = f"<b>Point {idx + 1}</b><br>"
        popup_text += f"Lat: {row['latitude']:.6f}<br>"
        popup_text += f"Lon: {row['longitude']:.6f}<br>"
        
        if 'stop_name' in row and pd.notna(row['stop_name']) and str(row['stop_name']).strip():
            popup_text += f"Stop: {row['stop_name']}<br>"
        
        if 'route_number' in row and pd.notna(row['route_number']):
            popup_text += f"Route: {row['route_number']}<br>"
        
        if 'stop_number' in row and pd.notna(row['stop_number']):
            popup_text += f"Stop #: {row['stop_number']}<br>"
        
        if 'sequence' in row and pd.notna(row['sequence']):
            popup_text += f"Sequence: {row['sequence']}<br>"
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=row.get('stop_name', f"Point {idx + 1}"),
            icon=folium.Icon(color='blue', icon='map-pin', prefix='fa')
        ).add_to(m)
    
    return m

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['csv_file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and file.filename.lower().endswith('.csv'):
            try:
                # Read CSV file
                df = pd.read_csv(file)
                
                # Check required columns
                required_cols = ['latitude', 'longitude']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    flash(f'Missing required columns: {", ".join(missing_cols)}', 'error')
                    return redirect(request.url)
                
                # Validate coordinates
                try:
                    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
                    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
                    
                    # Remove rows with invalid coordinates
                    initial_count = len(df)
                    df = df.dropna(subset=['latitude', 'longitude'])
                    
                    if len(df) == 0:
                        flash('No valid coordinate data found', 'error')
                        return redirect(request.url)
                    
                    if len(df) < initial_count:
                        flash(f'Removed {initial_count - len(df)} rows with invalid coordinates', 'info')
                
                except ValueError:
                    flash('Invalid coordinate data format', 'error')
                    return redirect(request.url)
                
                # Check coordinate ranges
                invalid_lat = df[(df['latitude'] < -90) | (df['latitude'] > 90)]
                invalid_lon = df[(df['longitude'] < -180) | (df['longitude'] > 180)]
                
                if len(invalid_lat) > 0 or len(invalid_lon) > 0:
                    flash('Some coordinates are outside valid ranges (lat: -90 to 90, lon: -180 to 180)', 'error')
                    return redirect(request.url)
                
                # Create map
                map_obj = create_map(df)
                if map_obj is None:
                    flash('Could not create map from data', 'error')
                    return redirect(request.url)
                
                # Generate statistics
                routes_count = 0
                if 'route_number' in df.columns:
                    routes_count = df['route_number'].nunique()
                
                stops_with_names = 0
                if 'stop_name' in df.columns:
                    stops_with_names = df['stop_name'].notna().sum()
                
                # Convert map to HTML
                map_html = map_obj._repr_html_()
                
                flash(f'Successfully loaded {len(df)} coordinate points', 'success')
                
                return render_template_string(
                    MAIN_TEMPLATE,
                    data=df.to_dict('records'),
                    map_html=map_html,
                    routes_count=routes_count,
                    stops_with_names=stops_with_names
                )
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Please upload a CSV file', 'error')
            return redirect(request.url)
    
    return render_template_string(MAIN_TEMPLATE)

if __name__ == '__main__':
    print("CSV Coordinate Viewer")
    print("=" * 30)
    print("Starting web server...")
    print("Open your browser and go to:")
    print("  - Local access: http://localhost:5000")
    print("  - Network access: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        app.run(debug=True, host='127.0.0.1', port=5000)
    except OSError as e:
        if "Address already in use" in str(e):
            print("Port 5000 is already in use. Trying port 5001...")
            try:
                app.run(debug=True, host='127.0.0.1', port=5001)
            except OSError:
                print("Port 5001 is also in use. Trying port 8080...")
                app.run(debug=True, host='127.0.0.1', port=8080)
        else:
            raise e
    print("CSV Coordinate Viewer")
    print("=" * 30)
    print("Starting web server...")
    print("Open your browser and go to:")
    print("  - Local access: http://localhost:5000")
    print("  - Network access: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        app.run(debug=True, host='127.0.0.1', port=5000)
    except OSError as e:
        if "Address already in use" in str(e):
            print("Port 5000 is already in use. Trying port 5001...")
            try:
                app.run(debug=True, host='127.0.0.1', port=5001)
            except OSError:
                print("Port 5001 is also in use. Trying port 8080...")
                app.run(debug=True, host='127.0.0.1', port=8080)
        else:
            raise e
