import pandas as pd
import numpy as np
import re

def extract_bus_count(count_str):
    """Extract bus count from strings like '13+1W', '20+2W', '10+W', etc."""
    if pd.isna(count_str) or count_str == '':
        return 0
    
    if isinstance(count_str, str):
        # Find all numbers and sum them (e.g., "20+2W" = 20+2 = 22)
        numbers = re.findall(r'\d+', count_str)
        if numbers:
            return sum(int(num) for num in numbers)
        else:
            return 0
    else:
        return int(count_str) if not pd.isna(count_str) else 0

def analyze_input_data():
    """Analyze the input files"""
    print("="*60)
    print("ANALYZING INPUT DATA")
    print("="*60)
    
    # Load snapped.csv
    df_snapped = pd.read_csv('snapped.csv')
    print(f"Snapped.csv:")
    print(f"- Total stops: {len(df_snapped)}")
    print(f"- Total students: {df_snapped['num_students'].sum()}")
    print(f"- Columns: {list(df_snapped.columns)}")
    print(f"- Student range: {df_snapped['num_students'].min()} to {df_snapped['num_students'].max()}")
    
    # Load coords.csv
    df_coords = pd.read_csv('coords.csv')
    print(f"\nCoords.csv:")
    print(f"- Total locations: {len(df_coords)}")
    print(f"- Columns: {list(df_coords.columns)}")
    
    # Analyze depot bus counts
    print(f"\nDepot Analysis:")
    total_buses = 0
    active_depots = 0
    for idx, row in df_coords.iterrows():
        bus_count = extract_bus_count(row['Counts'])
        if bus_count > 0:
            active_depots += 1
            total_buses += bus_count
            print(f"  {row['Parking Name']}: {bus_count} buses")
    
    print(f"\nSummary:")
    print(f"- Active depots: {active_depots}")
    print(f"- Total buses available: {total_buses}")
    print(f"- Student pickup stops: {len(df_snapped)}")
    print(f"- Total students: {df_snapped['num_students'].sum()}")
    print(f"- Students per bus (if all used): {df_snapped['num_students'].sum() / total_buses:.1f}")
    
    return df_snapped, df_coords

def create_mdvrp_dataset():
    """Create MDVRP dataset by merging snapped.csv and coords.csv"""
    print(f"\n{'='*60}")
    print("CREATING MDVRP DATASET")
    print("="*60)
    
    # Load data
    df_snapped = pd.read_csv('snapped.csv')
    df_coords = pd.read_csv('coords.csv')
    
    mdvrp_data = []
    
    # Step 1: Add depots from coords.csv
    depot_count = 0
    depot_bus_allocation = {}
    
    for idx, row in df_coords.iterrows():
        bus_count = extract_bus_count(row['Counts'])
        
        # Only include depots with buses
        if bus_count > 0:
            depot_count += 1
            depot_bus_allocation[depot_count-1] = bus_count  # 0-indexed for later use
            
            mdvrp_data.append({
                'stop_id': f'D{depot_count}',
                'stop_name': row['Parking Name'],
                'latitude': row['Latitude'],
                'longitude': row['Longitude'],
                'num_students': 0,
                'is_depot': True,
                'is_school': False
            })
    
    print(f"Added {depot_count} depots")
    for i, (depot_idx, buses) in enumerate(depot_bus_allocation.items()):
        print(f"  D{i+1}: {mdvrp_data[i]['stop_name']} - {buses} buses")
    
    # Step 2: Add student pickup stops from snapped.csv
    for idx, row in df_snapped.iterrows():
        stop_number = idx + 1  # Start from S1
        
        mdvrp_data.append({
            'stop_id': f'S{stop_number}',
            'stop_name': f'Pickup Stop {stop_number}',  # You can customize this
            'latitude': row['latitude'],  # Use snapped latitude
            'longitude': row['longitude'],  # Use snapped longitude
            'num_students': row['num_students'],
            'is_depot': False,
            'is_school': False
        })
    
    print(f"Added {len(df_snapped)} student pickup stops")
    
    # Step 3: Add school (end point)
    # Using a representative location in Chennai (you can modify this)
    school_lat = 12.9236  # Average of your data points
    school_lon = 80.1437  # Average of your data points
    
    mdvrp_data.append({
        'stop_id': 'E1',
        'stop_name': 'Main School (End Point)',
        'latitude': school_lat,
        'longitude': school_lon,
        'num_students': 0,
        'is_depot': False,
        'is_school': True
    })
    
    print(f"Added 1 school destination")
    
    # Create DataFrame
    df_mdvrp = pd.DataFrame(mdvrp_data)
    
    # Save to CSV in exact format as mdvrp_data.csv
    df_mdvrp.to_csv('merged_mdvrp_data.csv', index=False)
    
    print(f"\n{'='*60}")
    print("MDVRP DATASET CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"File saved: merged_mdvrp_data.csv")
    print(f"Total locations: {len(df_mdvrp)}")
    print(f"- Depots: {len(df_mdvrp[df_mdvrp['is_depot']])}")
    print(f"- Pickup stops: {len(df_mdvrp[(df_mdvrp['is_depot'] == False) & (df_mdvrp['is_school'] == False)])}")
    print(f"- Schools: {len(df_mdvrp[df_mdvrp['is_school']])}")
    print(f"- Total students: {df_mdvrp['num_students'].sum()}")
    
    # Save depot allocation info for MDVRP solver
    with open('depot_bus_allocation.py', 'w') as f:
        f.write("# Depot bus allocation for MDVRP solver\n")
        f.write("# Use this in your MDVRP solver configuration\n\n")
        f.write("depot_bus_allocation = {\n")
        for depot_idx, bus_count in depot_bus_allocation.items():
            depot_name = df_mdvrp[df_mdvrp['stop_id'] == f'D{depot_idx+1}'].iloc[0]['stop_name']
            f.write(f"    {depot_idx}: {bus_count},  # {depot_name}\n")
        f.write("}\n\n")
        f.write(f"# Configuration summary:\n")
        f.write(f"# Total buses: {sum(depot_bus_allocation.values())}\n")
        f.write(f"# Total students: {df_mdvrp['num_students'].sum()}\n")
        f.write(f"# Bus capacity: 55  # Set as needed\n")
        f.write(f"# Total vehicles: {sum(depot_bus_allocation.values())}\n")
    
    print(f"Depot allocation saved: depot_bus_allocation.py")
    
    return df_mdvrp, depot_bus_allocation

def verify_output(df_mdvrp):
    """Verify the output matches mdvrp_data.csv format"""
    print(f"\n{'='*60}")
    print("VERIFYING OUTPUT FORMAT")
    print("="*60)
    
    required_columns = ['stop_id', 'stop_name', 'latitude', 'longitude', 'num_students', 'is_depot', 'is_school']
    
    print("Column verification:")
    for col in required_columns:
        if col in df_mdvrp.columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ✗ {col} - MISSING!")
    
    print(f"\nData types verification:")
    print(f"  stop_id: {df_mdvrp['stop_id'].dtype}")
    print(f"  stop_name: {df_mdvrp['stop_name'].dtype}")
    print(f"  latitude: {df_mdvrp['latitude'].dtype}")
    print(f"  longitude: {df_mdvrp['longitude'].dtype}")
    print(f"  num_students: {df_mdvrp['num_students'].dtype}")
    print(f"  is_depot: {df_mdvrp['is_depot'].dtype}")
    print(f"  is_school: {df_mdvrp['is_school'].dtype}")
    
    print(f"\nSample of generated data:")
    print(df_mdvrp.head(10).to_string(index=False))
    
    print(f"\nDepots:")
    depots = df_mdvrp[df_mdvrp['is_depot'] == True]
    print(depots[['stop_id', 'stop_name', 'latitude', 'longitude']].to_string(index=False))
    
    print(f"\nSchool:")
    school = df_mdvrp[df_mdvrp['is_school'] == True]
    print(school[['stop_id', 'stop_name', 'latitude', 'longitude']].to_string(index=False))

def main():
    """Main function"""
    try:
        # Analyze input data
        df_snapped, df_coords = analyze_input_data()
        
        # Create MDVRP dataset
        df_mdvrp, depot_allocation = create_mdvrp_dataset()
        
        # Verify output format
        verify_output(df_mdvrp)
        
        print(f"\n{'='*60}")
        print("CONVERSION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Generated files:")
        print("1. merged_mdvrp_data.csv - Main MDVRP dataset")
        print("2. depot_bus_allocation.py - Depot configuration for solver")
        
        print(f"\nNext steps:")
        print("1. Use 'merged_mdvrp_data.csv' as input to your MDVRP solver")
        print("2. Import depot allocation from 'depot_bus_allocation.py'")
        print("3. Set bus capacity to 55 in your solver")
        
        # Show summary statistics
        total_students = df_mdvrp['num_students'].sum()
        total_buses = sum(depot_allocation.values())
        
        print(f"\nFinal Statistics:")
        print(f"- Total locations: {len(df_mdvrp)}")
        print(f"- Total students: {total_students}")
        print(f"- Total buses: {total_buses}")
        print(f"- Students per bus: {total_students / total_buses:.1f}")
        print(f"- Capacity utilization (55/bus): {(total_students / (total_buses * 55)) * 100:.1f}%")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please ensure 'snapped.csv' and 'coords.csv' are in the current directory")
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()