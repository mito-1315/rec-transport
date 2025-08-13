import pandas as pd
import folium
from folium import plugins
import os
from datetime import datetime

def create_simple_clustered_visualization():
    """Create a simple, clean clustered visualization with individual day/time slot toggling"""
    
    print("🚀 Creating simple clustered route visualization...")
    
    # College coordinates (center point)
    college_lat, college_lon = 13.008794724595475, 80.00342657961114
    
    # Create base map
    m = folium.Map(
        location=[college_lat, college_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
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
    
    # Create feature groups for each day/time combination
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    time_slots = ['8_am', '10_am', '3_pm', '5_pm', 'Leave']
    
    # Create feature groups for each day/time combination
    feature_groups = {}
    
    for day in days:
        for time_slot in time_slots:
            group_name = f"{day} - {time_slot.replace('_', ' ').title()}"
            feature_groups[f"{day}_{time_slot}"] = folium.FeatureGroup(
                name=group_name, 
                show=False  # Start with all hidden
            )
            m.add_child(feature_groups[f"{day}_{time_slot}"])
    
    # Process each day and time slot
    total_students = 0
    total_centroids = 0
    
    for day in days:
        print(f"📅 Processing {day}...")
        
        for time_slot in time_slots:
            time_color = time_colors.get(time_slot, 'gray')
            group_key = f"{day}_{time_slot}"
            feature_group = feature_groups[group_key]
            
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
                
                # Add centroid markers (large circles for clusters)
                for idx, centroid in centroids_df.iterrows():
                    # Calculate marker size based on student count
                    student_count = centroid.get('num_students', 0)
                    total_students += student_count
                    marker_size = max(12, min(35, 8 + int(student_count) * 3))
                    
                    popup_text = f"""
                    <b>📍 {day} {time_slot.replace('_', ' ').title()} - Cluster {idx+1}</b><br>
                    <b>Students:</b> {int(student_count)}<br>
                    <b>Location:</b> {centroid.get('route_name', 'Unknown')}<br>
                    <b>Coordinates:</b> {centroid.get('snapped_lat', 0):.4f}, {centroid.get('snapped_lon', 0):.4f}
                    """
                    
                    # Create centroid marker (large circle representing the cluster)
                    folium.CircleMarker(
                        [centroid['snapped_lat'], centroid['snapped_lon']],
                        radius=marker_size,
                        popup=folium.Popup(popup_text, max_width=300),
                        color=time_color,
                        fillColor=time_color,
                        fillOpacity=0.8,
                        weight=3
                    ).add_to(feature_group)
                    
                    # Add cluster label with student count
                    folium.Marker(
                        [centroid['snapped_lat'], centroid['snapped_lon']],
                        popup=f"<b>Cluster {idx+1}</b><br>Students: {int(student_count)}",
                        icon=folium.DivIcon(
                            html=f'<div style="background-color: white; border: 2px solid {time_color}; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: {time_color}; font-size: 12px;">{int(student_count)}</div>',
                            icon_size=(30, 30),
                            icon_anchor=(15, 15)
                        )
                    ).add_to(feature_group)
            
            # Load assignments to show individual students within clusters
            assignments_file = f"{base_path}/{time_slot}_assignments.csv"
            if os.path.exists(assignments_file):
                assignments_df = pd.read_csv(assignments_file)
                
                # Add individual student markers (small dots within clusters)
                for idx, assignment in assignments_df.iterrows():
                    if idx < 150:  # Limit for performance
                        student_popup = f"""
                        <b>👤 Student</b><br>
                        <b>Day:</b> {day}<br>
                        <b>Time:</b> {time_slot.replace('_', ' ').title()}<br>
                        <b>Department:</b> {assignment.get('department', 'Unknown')}<br>
                        <b>Distance to Cluster:</b> {assignment.get('road_distance_km', 0):.2f} km
                        """
                        
                        folium.CircleMarker(
                            [assignment['student_lat'], assignment['student_lon']],
                            radius=2,
                            popup=folium.Popup(student_popup, max_width=250),
                            color=time_color,
                            fillColor=time_color,
                            fillOpacity=0.4,
                            weight=1
                        ).add_to(feature_group)
    
    # Add layer control
    folium.LayerControl(
        collapsed=False,
        position='topright'
    ).add_to(m)
    
    # Add measure tool
    plugins.MeasureControl().add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    # Add comprehensive legend
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 400px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                overflow-y: auto; max-height: 500px;">
    <b style="font-size: 14px; color: #333;">🗺️ Clean Clustered Route Visualization</b><br><br>
    
    <b>📊 Summary:</b><br>
    <div style="margin-left: 10px;">
        • Total Students: {total_students}<br>
        • Total Clusters: {total_centroids}<br>
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
        • <span style="color: blue;">●</span> Large circles = Clusters (centroids)<br>
        • <span style="color: blue;">●</span> Small dots = Individual students<br>
        • Numbers in circles = Student count per cluster<br>
        • Circle size = Number of students
    </div><br>
    
    <b>🔧 How to Use:</b><br>
    <div style="margin-left: 10px;">
        • Use layer control (top right) to toggle individual day/time slots<br>
        • Each checkbox = one day and time slot combination<br>
        • Click on clusters for detailed information<br>
        • Use measure tool for distances
    </div>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save the map
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"simple_clustered_routes_{timestamp}.html"
    
    m.save(output_file)
    print(f"✅ Simple clustered visualization saved to: {output_file}")
    print(f"📊 Total Students: {total_students}")
    print(f"📊 Total Clusters: {total_centroids}")
    
    return output_file

if __name__ == "__main__":
    print("🚀 Starting simple clustered route visualization...")
    
    # Create visualization
    output_file = create_simple_clustered_visualization()
    
    if output_file:
        print(f"\n✅ SUCCESS! Simple clustered visualization created!")
        print(f"📁 File: {output_file}")
        
        print("\n🚀 Opening in browser...")
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(output_file)}")
        
        print("\n🎯 FEATURES:")
        print("   ✅ Clean clustered visualization")
        print("   ✅ Individual day/time slot toggling")
        print("   ✅ Large circles = Clusters (centroids)")
        print("   ✅ Small dots = Individual students")
        print("   ✅ Student count labels on clusters")
        print("   ✅ Layer control for easy toggling")
        print("   ✅ Click clusters for detailed information")
    else:
        print("❌ Failed to create visualization") 