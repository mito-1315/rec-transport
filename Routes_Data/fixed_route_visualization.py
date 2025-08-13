import pandas as pd
import folium
from folium import plugins
import os
from datetime import datetime
import glob

def create_fixed_route_visualization():
    """Create a fixed route visualization that correctly reads existing data"""
    
    print("🚀 Creating fixed route visualization...")
    
    # College coordinates (center point)
    college_lat, college_lon = 13.008794724595475, 80.00342657961114
    
    # Create base map
    m = folium.Map(
        location=[college_lat, college_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add multiple tile layers
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>',
        name='Terrain'
    ).add_to(m)
    
    folium.TileLayer(
        tiles='CartoDB positron',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        name='Light'
    ).add_to(m)
    
    # Add college marker
    folium.Marker(
        [college_lat, college_lon],
        popup="<b>🏫 College - Starting Point</b>",
        icon=folium.Icon(color='red', icon='graduation-cap', prefix='fa')
    ).add_to(m)
    
    # Colors for different time slots
    time_colors = {
        '8_am': 'blue',
        '10_am': 'green', 
        '3_pm': 'orange',
        '5_pm': 'purple',
        'Leave': 'darkred'
    }
    
    # Create feature groups for each day
    day_groups = {}
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    
    for day in days:
        day_groups[day] = folium.FeatureGroup(name=f'{day}', show=True)
        m.add_child(day_groups[day])
    
    # Create feature groups for each time slot
    time_groups = {}
    time_slots = ['8_am', '10_am', '3_pm', '5_pm', 'Leave']
    
    for time_slot in time_slots:
        time_groups[time_slot] = folium.FeatureGroup(name=f'{time_slot}', show=True)
        m.add_child(time_groups[time_slot])
    
    # Create all points group for search
    all_points_group = folium.FeatureGroup(name='All Points', show=True)
    m.add_child(all_points_group)
    
    total_students = 0
    total_centroids = 0
    
    # Process each day
    for day in days:
        print(f"📅 Processing {day}...")
        
        # Process each time slot
        for time_slot in time_slots:
            time_color = time_colors.get(time_slot, 'gray')
            
            # Check if files exist for this time slot
            base_path = f"{day}/{time_slot}"
            if not os.path.exists(base_path):
                continue
                
            print(f"  ⏰ Processing {time_slot}...")
            
            # Load centroids (snapped)
            centroids_file = f"{base_path}/{time_slot}_centroids_snapped.csv"
            if os.path.exists(centroids_file):
                centroids_df = pd.read_csv(centroids_file)
                total_centroids += len(centroids_df)
                
                # Add centroid markers
                for idx, centroid in centroids_df.iterrows():
                    # Calculate marker size based on student count
                    student_count = centroid.get('num_students', 0)
                    total_students += student_count
                    marker_size = max(8, min(25, 5 + int(student_count) * 2))
                    
                    popup_text = f"""
                    <b>📍 {day} {time_slot} - Centroid {idx+1}</b><br>
                    <b>Students:</b> {int(student_count)}<br>
                    <b>Location:</b> {centroid.get('route_name', 'Unknown')}<br>
                    <b>Coordinates:</b> {centroid.get('snapped_lat', 0):.4f}, {centroid.get('snapped_lon', 0):.4f}<br>
                    <b>Snap Distance:</b> {centroid.get('snap_distance_meters', 0):.1f}m
                    """
                    
                    circle_marker = folium.CircleMarker(
                        [centroid['snapped_lat'], centroid['snapped_lon']],
                        radius=marker_size,
                        popup=folium.Popup(popup_text, max_width=300),
                        color=time_color,
                        fillColor=time_color,
                        fillOpacity=0.7,
                        weight=2
                    )
                    
                    # Add to all relevant groups
                    circle_marker.add_to(m)
                    circle_marker.add_to(day_groups[day])
                    circle_marker.add_to(time_groups[time_slot])
                    circle_marker.add_to(all_points_group)
            
            # Load assignments to see student distribution (limit for performance)
            assignments_file = f"{base_path}/{time_slot}_assignments.csv"
            if os.path.exists(assignments_file):
                assignments_df = pd.read_csv(assignments_file)
                
                # Add individual student markers (smaller, more transparent) - limit to first 50 per time slot
                for idx, assignment in assignments_df.iterrows():
                    if idx < 50:  # Limit for performance
                        student_popup = f"""
                        <b>👤 Student Assignment</b><br>
                        <b>Day:</b> {day}<br>
                        <b>Time:</b> {time_slot}<br>
                        <b>Department:</b> {assignment.get('department', 'Unknown')}<br>
                        <b>Student Coordinates:</b> {assignment.get('student_lat', 0):.4f}, {assignment.get('student_lon', 0):.4f}<br>
                        <b>Distance to Centroid:</b> {assignment.get('road_distance_km', 0):.2f} km
                        """
                        
                        student_marker = folium.CircleMarker(
                            [assignment['student_lat'], assignment['student_lon']],
                            radius=3,
                            popup=folium.Popup(student_popup, max_width=250),
                            color=time_color,
                            fillColor=time_color,
                            fillOpacity=0.3,
                            weight=1
                        )
                        
                        # Add to all relevant groups
                        student_marker.add_to(m)
                        student_marker.add_to(day_groups[day])
                        student_marker.add_to(time_groups[time_slot])
                        student_marker.add_to(all_points_group)
    
    # Add layer control
    folium.LayerControl(
        collapsed=False,
        position='topright'
    ).add_to(m)
    
    # Add search functionality
    search = plugins.Search(
        layer=all_points_group,
        geom_type='Point',
        placeholder='Search locations...',
        collapsed=False,
    )
    m.add_child(search)
    
    # Add measure tool
    plugins.MeasureControl().add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    # Add comprehensive legend
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 350px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                overflow-y: auto; max-height: 400px;">
    <b style="font-size: 14px; color: #333;">🗺️ Route Data Visualization</b><br><br>
    
    <b>📊 Summary:</b><br>
    <div style="margin-left: 10px;">
        • Total Students: {total_students}<br>
        • Total Centroids: {total_centroids}<br>
        • Days: {len(days)}<br>
        • Time Slots: {len(time_slots)}
    </div><br>
    
    <b>⏰ Time Slots:</b><br>
    <div style="margin-left: 10px;">
        • <span style="color: blue;">● 8:00 AM</span> - Morning pickup<br>
        • <span style="color: green;">● 10:00 AM</span> - Mid-morning<br>
        • <span style="color: orange;">● 3:00 PM</span> - Afternoon<br>
        • <span style="color: purple;">● 5:00 PM</span> - Evening<br>
        • <span style="color: darkred;">● Leave</span> - Departure
    </div><br>
    
    <b>📍 Map Elements:</b><br>
    <div style="margin-left: 10px;">
        • <i class="fa fa-graduation-cap" style="color:red"></i> College Start Point<br>
        • <span style="color: blue;">●</span> Large circles = Centroids (grouped stops)<br>
        • <span style="color: blue;">●</span> Small dots = Individual students<br>
        • Circle size = Number of students
    </div><br>
    
    <b>🔧 Tools:</b><br>
    <div style="margin-left: 10px;">
        • Layer control (top right) - Toggle days/time slots<br>
        • Search box for finding locations<br>
        • Measure tool for distances<br>
        • Fullscreen mode
    </div>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save the map
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"fixed_routes_map_{timestamp}.html"
    
    m.save(output_file)
    print(f"✅ Fixed route visualization saved to: {output_file}")
    print(f"📊 Total Students: {total_students}")
    print(f"📊 Total Centroids: {total_centroids}")
    
    return output_file

if __name__ == "__main__":
    print("🚀 Starting fixed route visualization...")
    
    # Create visualization
    output_file = create_fixed_route_visualization()
    
    if output_file:
        print(f"\n✅ SUCCESS! Fixed visualization created!")
        print(f"📁 File: {output_file}")
        
        print("\n🚀 Opening in browser...")
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(output_file)}")
        
        print("\n🎯 FEATURES:")
        print("   ✅ All days and time slots visualized")
        print("   ✅ Centroids (grouped stops) shown as large circles")
        print("   ✅ Individual students shown as small dots")
        print("   ✅ Color-coded by time slot")
        print("   ✅ Layer control to toggle days/time slots")
        print("   ✅ Click markers for detailed information")
        print("   ✅ Search functionality")
        print("   ✅ Measure tool")
        print("   ✅ Comprehensive legend")
    else:
        print("❌ Failed to create visualization") 