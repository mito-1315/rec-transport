#!/usr/bin/env python3
"""
Extract No Coordinates Script
Extracts all stop names that have no latitude or longitude coordinates
from the optimized coordinates CSV file.
"""

import pandas as pd
import json
import os
from pathlib import Path

def extract_stops_without_coordinates(csv_file_path, output_file_path):
    """
    Extract stop names that have no coordinates from CSV file.
    
    Args:
        csv_file_path: Path to the input CSV file
        output_file_path: Path to the output JSON file
        
    Returns:
        dict: Summary of extraction results
    """
    try:
        # Read the CSV file
        print(f"Reading CSV file: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        total_records = len(df)
        print(f"Total records in CSV: {total_records}")
        
        # Find rows where latitude or longitude is missing/empty
        no_coords_mask = (
            df['latitude'].isna() | 
            df['longitude'].isna() | 
            (df['latitude'] == '') | 
            (df['longitude'] == '')
        )
        
        # Filter rows without coordinates
        no_coords_df = df[no_coords_mask]
        
        records_without_coords = len(no_coords_df)
        print(f"Records without coordinates: {records_without_coords}")
        
        if records_without_coords == 0:
            print("No records found without coordinates!")
            result_list = []
        else:
            # Get unique stop names only
            unique_stops = no_coords_df['stop_name'].unique()
            unique_stops = [stop for stop in unique_stops if pd.notna(stop) and str(stop).strip()]
            
            # Sort alphabetically
            result_list = sorted(unique_stops)
            
            print(f"Unique stops without coordinates: {len(result_list)}")
            print(f"\nFirst 10 stops without coordinates:")
            for i, stop in enumerate(result_list[:10]):
                print(f"{i+1:2d}. {stop}")
            
            if len(result_list) > 10:
                print(f"... and {len(result_list) - 10} more stops")
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write to JSON file as simple array
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result_list, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file_path}")
        
        # Return summary dictionary
        return {
            'total_stops_in_file': total_records,
            'stops_without_coords': records_without_coords,
            'unique_stops_without_coords': len(result_list),
            'stop_list': result_list
        }
        
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def main():
    """Main function to run the extraction"""
    # Set up file paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent  # Go up to ApproachToBusNetwork
    
    csv_file_path = project_root / "rec-transport" / "output" / "mtc_optimized_coordinates.csv"
    output_file_path = project_root / "rec-transport" / "output" / "mtc_no_coords.json"
    
    print("=" * 60)
    print("MTC Stops Without Coordinates Extractor")
    print("=" * 60)
    print(f"Input file: {csv_file_path}")
    print(f"Output file: {output_file_path}")
    print()
    
    # Check if input file exists
    if not csv_file_path.exists():
        print(f"Error: Input CSV file does not exist: {csv_file_path}")
        print("Please make sure the mtc_optimized_coordinates.csv file is in the output folder.")
        return
    
    # Extract stops without coordinates
    result = extract_stops_without_coordinates(csv_file_path, output_file_path)
    
    if result is not None:
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total records processed: {result['total_stops_in_file']}")
        print(f"Records without coordinates: {result['stops_without_coords']}")
        print(f"Unique stops without coordinates: {result['unique_stops_without_coords']}")
        print(f"Results saved to: {output_file_path}")
    else:
        print("Extraction failed!")

if __name__ == "__main__":
    main()