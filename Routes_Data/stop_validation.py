import pandas as pd
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns

class SimpleBusStopValidator:
    """
    Basic validation without external APIs - uses geodesic distance approximation
    """
    
    def __init__(self):
        self.validation_results = {}
    
    def load_and_validate_data(self, day='Friday', time_slot='3_pm'):
        """Load data for a specific day and time slot and perform basic validation"""
        
        try:
            # Construct file paths based on your structure
            base_path = f"{day}/{time_slot}/"
            assignments = pd.read_csv(f'{base_path}{time_slot}_assignments.csv')
            centroids_snapped = pd.read_csv(f'{base_path}{time_slot}_centroids_snapped.csv')
            centroids_original = pd.read_csv(f'{base_path}{time_slot}_centroids.csv')
            
            print(f"Loaded data for {day} {time_slot}:")
            print(f"  - Assignments: {len(assignments)} records")
            print(f"  - Snapped centroids: {len(centroids_snapped)} records")
            print(f"  - Original centroids: {len(centroids_original)} records")
            
            return assignments, centroids_snapped, centroids_original
            
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            print(f"Make sure you're running this script from the Routes_Data/ directory")
            return None, None, None
    
    def validate_1km_distance_approximation(self, assignments, bus_stops):
        """
        Validate 1km constraint using geodesic distance (road distance approximation)
        Geodesic * 1.3 ≈ road distance for urban areas
        """
        results = []
        road_factor = 1.3  # Approximation factor for road vs straight-line distance
        
        # Use the actual column names from the data files
        student_lat_col = 'student_lat'  # From assignments file
        student_lon_col = 'student_lon'  # From assignments file
        stop_lat_col = 'snapped_lat'     # From centroids_snapped file
        stop_lon_col = 'snapped_lon'     # From centroids_snapped file
        
        # Verify columns exist
        if student_lat_col not in assignments.columns:
            print(f"❌ Column '{student_lat_col}' not found in assignments. Available: {list(assignments.columns)}")
            return []
        if stop_lat_col not in bus_stops.columns:
            print(f"❌ Column '{stop_lat_col}' not found in bus_stops. Available: {list(bus_stops.columns)}")
            return []
        
        print(f"✅ Using columns: students({student_lat_col}, {student_lon_col}), stops({stop_lat_col}, {stop_lon_col})")
        
        for idx, student in assignments.iterrows():
            try:
                student_coords = (student[student_lat_col], student[student_lon_col])
                
                # Find assigned bus stop using cluster_id
                stop_id = student.get('cluster_id')
                
                if pd.isna(stop_id):
                    results.append({
                        'student_id': student.get('user', idx),
                        'student_lat': student_coords[0],
                        'student_lon': student_coords[1],
                        'stop_id': None,
                        'geodesic_distance_km': None,
                        'estimated_road_distance_km': None,
                        'within_1km_road': False,
                        'status': 'No assigned stop'
                    })
                    continue
                
                # Find the bus stop using cluster_number
                bus_stop = bus_stops[bus_stops['cluster_number'] == stop_id]
                
                if bus_stop.empty:
                    results.append({
                        'student_id': student.get('user', idx),
                        'student_lat': student_coords[0],
                        'student_lon': student_coords[1],
                        'stop_id': stop_id,
                        'geodesic_distance_km': None,
                        'estimated_road_distance_km': None,
                        'within_1km_road': False,
                        'status': 'Bus stop not found'
                    })
                    continue
                
                stop_coords = (bus_stop.iloc[0][stop_lat_col], bus_stop.iloc[0][stop_lon_col])
                geodesic_dist = geodesic(student_coords, stop_coords).kilometers
                estimated_road_dist = geodesic_dist * road_factor
                
                results.append({
                    'student_id': student.get('student_id', idx),
                    'student_lat': student_coords[0],
                    'student_lon': student_coords[1],
                    'stop_id': stop_id,
                    'stop_lat': stop_coords[0],
                    'stop_lon': stop_coords[1],
                    'geodesic_distance_km': geodesic_dist,
                    'estimated_road_distance_km': estimated_road_dist,
                    'within_1km_road': estimated_road_dist <= 1.0,
                    'status': 'Valid' if estimated_road_dist <= 1.0 else f'Over limit: {estimated_road_dist:.2f}km'
                })
                
            except Exception as e:
                print(f"Error processing student {idx}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def analyze_snap_quality(self, original_centroids, snapped_centroids):
        """Analyze how much centroids moved during snapping"""
        snap_analysis = []
        
        # Match centroids by ID
        for idx, orig in original_centroids.iterrows():
            # Find corresponding snapped centroid
            orig_id = orig.get('stop_id') or orig.get('id') or orig.get('cluster_id') or idx
            
            snapped = snapped_centroids[snapped_centroids.get('stop_id', snapped_centroids.get('id', snapped_centroids.index)) == orig_id]
            
            if snapped.empty:
                continue
            
            orig_coords = (orig['latitude'], orig['longitude'])
            snapped_coords = (snapped.iloc[0]['latitude'], snapped.iloc[0]['longitude'])
            
            snap_distance = geodesic(orig_coords, snapped_coords).meters
            
            snap_analysis.append({
                'centroid_id': orig_id,
                'original_lat': orig_coords[0],
                'original_lon': orig_coords[1],
                'snapped_lat': snapped_coords[0],
                'snapped_lon': snapped_coords[1],
                'snap_distance_m': snap_distance,
                'significant_move': snap_distance > 200  # Flag moves > 200m
            })
        
        return pd.DataFrame(snap_analysis)
    
    def generate_validation_summary(self, distance_validation, snap_analysis):
        """Generate summary statistics"""
        
        summary = {
            'distance_validation': {
                'total_students': len(distance_validation),
                'within_1km': len(distance_validation[distance_validation['within_1km_road'] == True]),
                'over_1km': len(distance_validation[distance_validation['within_1km_road'] == False]),
                'no_assignment': len(distance_validation[distance_validation['stop_id'].isna()]),
                'compliance_rate': len(distance_validation[distance_validation['within_1km_road'] == True]) / len(distance_validation) * 100,
                'avg_distance_km': distance_validation['estimated_road_distance_km'].mean(),
                'max_distance_km': distance_validation['estimated_road_distance_km'].max(),
                'std_distance_km': distance_validation['estimated_road_distance_km'].std()
            },
            'snap_analysis': {
                'total_centroids': len(snap_analysis),
                'avg_snap_distance_m': snap_analysis['snap_distance_m'].mean(),
                'max_snap_distance_m': snap_analysis['snap_distance_m'].max(),
                'significant_moves': len(snap_analysis[snap_analysis['significant_move'] == True]),
                'significant_moves_pct': len(snap_analysis[snap_analysis['significant_move'] == True]) / len(snap_analysis) * 100
            }
        }
        
        return summary
    
    def create_visualizations(self, distance_validation, snap_analysis, day, time_slot):
        """Create validation visualizations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Bus Stop Validation Analysis - {day} {time_slot}', fontsize=16)
        
        # 1. Distance distribution
        axes[0, 0].hist(distance_validation['estimated_road_distance_km'].dropna(), 
                       bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='1km limit')
        axes[0, 0].set_xlabel('Estimated Road Distance (km)')
        axes[0, 0].set_ylabel('Number of Students')
        axes[0, 0].set_title('Student-to-Bus-Stop Distance Distribution')
        axes[0, 0].legend()
        
        # 2. Compliance pie chart
        compliance_data = distance_validation['within_1km_road'].value_counts()
        axes[0, 1].pie([compliance_data.get(True, 0), compliance_data.get(False, 0)], 
                      labels=['Within 1km', 'Over 1km'], 
                      autopct='%1.1f%%', 
                      colors=['lightgreen', 'lightcoral'])
        axes[0, 1].set_title('1km Distance Compliance')
        
        # 3. Snap distance distribution
        axes[1, 0].hist(snap_analysis['snap_distance_m'], 
                       bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(x=200, color='red', linestyle='--', linewidth=2, label='200m threshold')
        axes[1, 0].set_xlabel('Snap Distance (meters)')
        axes[1, 0].set_ylabel('Number of Centroids')
        axes[1, 0].set_title('Centroid Snapping Distance Distribution')
        axes[1, 0].legend()
        
        # 4. Distance vs compliance scatter
        valid_data = distance_validation.dropna(subset=['estimated_road_distance_km'])
        colors = ['green' if x else 'red' for x in valid_data['within_1km_road']]
        axes[1, 1].scatter(valid_data['estimated_road_distance_km'], 
                          range(len(valid_data)), 
                          c=colors, alpha=0.6, s=20)
        axes[1, 1].axvline(x=1.0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Estimated Road Distance (km)')
        axes[1, 1].set_ylabel('Student Index')
        axes[1, 1].set_title('Distance Compliance by Student')
        
        plt.tight_layout()
        
        # Save in validation_results directory
        import os
        os.makedirs('validation_results', exist_ok=True)
        plt.savefig(f'validation_results/{day}_{time_slot}_validation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def run_full_validation(self, day='Friday', time_slot='3_pm'):
        """Run complete validation for a specific day and time slot"""
        
        print(f"Starting validation for {day} {time_slot}...")
        
        # Load data
        assignments, centroids_snapped, centroids_original = self.load_and_validate_data(day, time_slot)
        
        if assignments is None:
            print("Could not load data. Please check file paths and ensure you're in Routes_Data/ directory.")
            return None
        
        # Validate distances
        print("Validating 1km distance constraint...")
        distance_validation = self.validate_1km_distance_approximation(assignments, centroids_snapped)
        
        # Analyze snapping
        print("Analyzing centroid snapping quality...")
        snap_analysis = self.analyze_snap_quality(centroids_original, centroids_snapped)
        
        # Generate summary
        summary = self.generate_validation_summary(distance_validation, snap_analysis)
        
        # Create visualizations
        print("Creating visualizations...")
        self.create_visualizations(distance_validation, snap_analysis, day, time_slot)
        
        # Create results directory structure
        import os
        results_dir = f'validation_results/{day}/{time_slot}/'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results
        distance_validation.to_csv(f'{results_dir}distance_validation.csv', index=False)
        snap_analysis.to_csv(f'{results_dir}snap_analysis.csv', index=False)
        
        # Save summary as JSON
        import json
        with open(f'{results_dir}validation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        print(f"\n=== VALIDATION SUMMARY FOR {day} {time_slot} ===")
        print(f"Distance Validation:")
        print(f"  - Total students: {summary['distance_validation']['total_students']}")
        print(f"  - Within 1km: {summary['distance_validation']['within_1km']} ({summary['distance_validation']['compliance_rate']:.1f}%)")
        print(f"  - Over 1km: {summary['distance_validation']['over_1km']}")
        print(f"  - Average distance: {summary['distance_validation']['avg_distance_km']:.3f} km")
        print(f"  - Maximum distance: {summary['distance_validation']['max_distance_km']:.3f} km")
        
        print(f"\nSnap Analysis:")
        print(f"  - Total centroids: {summary['snap_analysis']['total_centroids']}")
        print(f"  - Average snap distance: {summary['snap_analysis']['avg_snap_distance_m']:.1f} m")
        print(f"  - Maximum snap distance: {summary['snap_analysis']['max_snap_distance_m']:.1f} m")
        print(f"  - Significant moves (>200m): {summary['snap_analysis']['significant_moves']} ({summary['snap_analysis']['significant_moves_pct']:.1f}%)")
        
        # Identify problem cases
        problem_students = distance_validation[distance_validation['within_1km_road'] == False]
        if not problem_students.empty:
            print(f"\n⚠️  {len(problem_students)} students are over 1km from their assigned bus stop:")
            for _, student in problem_students.head(10).iterrows():  # Show first 10
                print(f"    Student {student['student_id']}: {student['estimated_road_distance_km']:.3f} km from stop {student['stop_id']}")
        
        return {
            'distance_validation': distance_validation,
            'snap_analysis': snap_analysis,
            'summary': summary
        }

# Usage
if __name__ == "__main__":
    validator = SimpleBusStopValidator()
    
    # Define available days and time slots based on your structure
    days = ['Friday', 'Monday', 'Wednesday', 'Saturday', 'Thursday', 'Tuesday']
    time_slots = ['3_pm', '5_pm', '8_am', '10_am', 'Leave']
    
    print("Bus Stop Validation Tool")
    print("Make sure you're running this from the Routes_Data/ directory")
    print("=" * 60)
    
    # Validate specific day/time combinations
    test_combinations = [
        ('Friday', '3_pm'),
        ('Friday', '5_pm'), 
        ('Friday', '8_am'),
        ('Wednesday', '3_pm')
    ]
    
    for day, slot in test_combinations:
        print(f"\n{'='*60}")
        print(f"VALIDATING {day.upper()} {slot.upper()}")
        print(f"{'='*60}")
        try:
            results = validator.run_full_validation(day, slot)
            if results:
                print(f"✅ Validation completed for {day} {slot}")
                print(f"Results saved in validation_results/{day}/{slot}/")
            else:
                print(f"❌ Validation failed for {day} {slot}")
        except Exception as e:
            print(f"❌ Error validating {day} {slot}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE")
    print("Check the validation_results/ directory for detailed reports")
    print("=" * 60)