#!/usr/bin/env python3
"""
Enhanced Script to clean CSV data by handling null values and geographic outliers.
This script processes all CSV files in the Routes_Data directory and provides options
for handling missing bus information and filtering out locations too far from Chennai.
"""

import os
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from pathlib import Path
import glob
from tqdm import tqdm
import datetime

# Chennai city center coordinates (approximate)
CHENNAI_CENTER = (13.009913, 80.002021)
MAX_REASONABLE_DISTANCE_KM = 65  # Maximum reasonable distance from city center

def is_valid_location(row):
    """
    Check if a location is within reasonable distance from Chennai city center.
    
    Args:
        row: DataFrame row containing 'latitude' and 'longitude' columns
        
    Returns:
        bool: True if location is within reasonable distance, False otherwise
    """
    try:
        # Check if coordinates are valid numbers
        lat = float(row['latitude'])
        lon = float(row['longitude'])
        
        # Check if coordinates are not zero or null
        if pd.isna(lat) or pd.isna(lon) or lat == 0 or lon == 0:
            return False
        
        # Calculate distance from Chennai center
        distance = geodesic(CHENNAI_CENTER, (lat, lon)).km
        return distance <= MAX_REASONABLE_DISTANCE_KM
        
    except (ValueError, TypeError):
        return False

def analyze_missing_data(csv_file):
    """
    Analyze missing data in a CSV file and return statistics.
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        dict: Statistics about missing data
    """
    df = pd.read_csv(csv_file)
    
    # Columns related to bus routing
    bus_columns = ['bus_no', 'boarding_point_name', 'pincode', 'latitude', 'longitude', 'address']
    
    stats = {
        'total_rows': len(df),
        'missing_data': {},
        'geographic_outliers': 0
    }
    
    for col in bus_columns:
        if col in df.columns:
            missing_count = df[col].isna().sum() + (df[col] == '').sum()
            missing_percentage = (missing_count / len(df)) * 100
            stats['missing_data'][col] = {
                'count': missing_count,
                'percentage': missing_percentage
            }
    
    # Count geographic outliers
    if 'latitude' in df.columns and 'longitude' in df.columns:
        valid_locations = df[df.apply(is_valid_location, axis=1)]
        stats['geographic_outliers'] = len(df) - len(valid_locations)
    
    return stats

def clean_csv_data(input_dir, output_dir=None, strategy='remove', filter_geographic=True, verbose=True, save_outliers=True):
    """
    Clean CSV data by handling null values and geographic outliers.
    
    Args:
        input_dir (str): Path to the input directory containing CSV files
        output_dir (str): Path to the output directory (optional, defaults to input_dir)
        strategy (str): Strategy for handling missing data
                       - 'remove': Remove rows with missing bus information
                       - 'keep': Keep all rows but mark missing data
                       - 'fill_default': Fill missing data with default values
        filter_geographic (bool): Whether to filter out locations too far from Chennai
        verbose (bool): Whether to print detailed progress information
        save_outliers (bool): Whether to save outliers data to separate files
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Convert to Path objects for easier handling
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all CSV files recursively
    csv_files = list(input_path.rglob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    if verbose:
        print(f"Found {len(csv_files)} CSV files to process:")
        for file in csv_files:
            print(f"  - {file}")
        print()
    
    processed_count = 0
    error_count = 0
    total_removed_missing = 0
    total_removed_geographic = 0
    error_files = []
    
    # Lists to collect all outliers data
    all_missing_data = []
    all_geographic_outliers = []
    
    # Use tqdm for progress bar if verbose is True
    file_iterator = tqdm(csv_files, desc="Processing files") if verbose else csv_files
    
    for csv_file in file_iterator:
        try:
            if verbose:
                print(f"Processing: {csv_file}")
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            original_count = len(df)
            
            # Columns related to bus routing
            bus_columns = ['bus_no', 'boarding_point_name', 'pincode', 'latitude', 'longitude', 'address']
            
            # Step 1: Handle missing data
            if strategy == 'remove':
                # Remove rows where any of the bus-related columns are missing
                missing_mask = df[bus_columns].isna().any(axis=1) | (df[bus_columns] == '').any(axis=1)
                df_cleaned = df[~missing_mask].copy()
                removed_missing = original_count - len(df_cleaned)
                total_removed_missing += removed_missing
                
                # Collect missing data for saving
                if save_outliers and removed_missing > 0:
                    missing_data = df[missing_mask].copy()
                    missing_data['source_file'] = str(csv_file.relative_to(input_path))
                    missing_data['removal_reason'] = 'missing_bus_data'
                    all_missing_data.append(missing_data)
                
                if verbose:
                    print(f"  ✓ Removed {removed_missing} rows with missing bus information")
                
            elif strategy == 'keep':
                # Keep all rows but add a flag column
                df_cleaned = df.copy()
                missing_mask = df[bus_columns].isna().any(axis=1) | (df[bus_columns] == '').any(axis=1)
                df_cleaned['has_missing_bus_data'] = missing_mask
                
                if verbose:
                    missing_count = missing_mask.sum()
                    print(f"  ✓ Marked {missing_count} rows with missing bus information")
                
            elif strategy == 'fill_default':
                # Fill missing values with default values
                df_cleaned = df.copy()
                
                # Fill missing values with appropriate defaults
                df_cleaned['bus_no'] = df_cleaned['bus_no'].fillna('UNKNOWN')
                df_cleaned['boarding_point_name'] = df_cleaned['boarding_point_name'].fillna('UNKNOWN')
                df_cleaned['pincode'] = df_cleaned['pincode'].fillna(0)
                df_cleaned['latitude'] = df_cleaned['latitude'].fillna(0.0)
                df_cleaned['longitude'] = df_cleaned['longitude'].fillna(0.0)
                df_cleaned['address'] = df_cleaned['address'].fillna('UNKNOWN')
                
                if verbose:
                    print(f"  ✓ Filled missing values with defaults")
            
            # Step 2: Filter geographic outliers
            if filter_geographic and 'latitude' in df_cleaned.columns and 'longitude' in df_cleaned.columns:
                before_geo_filter = len(df_cleaned)
                valid_locations = df_cleaned[df_cleaned.apply(is_valid_location, axis=1)]
                removed_geographic = before_geo_filter - len(valid_locations)
                total_removed_geographic += removed_geographic
                
                # Collect geographic outliers for saving
                if save_outliers and removed_geographic > 0:
                    geographic_outliers = df_cleaned[~df_cleaned.apply(is_valid_location, axis=1)].copy()
                    geographic_outliers['source_file'] = str(csv_file.relative_to(input_path))
                    geographic_outliers['removal_reason'] = 'geographic_outlier'
                    all_geographic_outliers.append(geographic_outliers)
                
                if verbose:
                    print(f"  ✓ Removed {removed_geographic} rows with geographic outliers")
                
                df_cleaned = valid_locations
            
            # Create the output path
            relative_path = csv_file.relative_to(input_path)
            output_file = output_path / relative_path
            
            # Create output directory if it doesn't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the cleaned data
            df_cleaned.to_csv(output_file, index=False)
            
            if verbose:
                print(f"  ✓ Saved: {output_file}")
            
            processed_count += 1
            
        except Exception as e:
            error_msg = f"Error processing {csv_file}: {str(e)}"
            error_files.append(str(csv_file.relative_to(input_path)))
            if verbose:
                print(f"  ✗ {error_msg}")
            error_count += 1
    
    # Save all outliers data to separate files
    if save_outliers:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save missing data outliers
        if all_missing_data:
            missing_outliers_file = output_path / f"all_missing_data_outliers_{timestamp}.csv"
            pd.concat(all_missing_data, ignore_index=True).to_csv(missing_outliers_file, index=False)
            if verbose:
                print(f"\n📁 All missing data outliers saved to: {missing_outliers_file}")
        
        # Save geographic outliers
        if all_geographic_outliers:
            geo_outliers_file = output_path / f"all_geographic_outliers_{timestamp}.csv"
            pd.concat(all_geographic_outliers, ignore_index=True).to_csv(geo_outliers_file, index=False)
            if verbose:
                print(f"📁 All geographic outliers saved to: {geo_outliers_file}")
        
        # Save combined outliers file
        if all_missing_data or all_geographic_outliers:
            combined_outliers_file = output_path / f"all_outliers_combined_{timestamp}.csv"
            all_outliers = []
            if all_missing_data:
                all_outliers.extend(all_missing_data)
            if all_geographic_outliers:
                all_outliers.extend(all_geographic_outliers)
            
            pd.concat(all_outliers, ignore_index=True).to_csv(combined_outliers_file, index=False)
            if verbose:
                print(f"📁 Combined outliers file saved to: {combined_outliers_file}")
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} files")
    if error_count > 0:
        print(f"Errors: {error_count} files")
        print("Files with errors:")
        for error_file in error_files:
            print(f"  - {error_file}")
    if strategy == 'remove':
        print(f"Total rows removed due to missing data: {total_removed_missing}")
    if filter_geographic:
        print(f"Total rows removed due to geographic outliers: {total_removed_geographic}")
    
    return processed_count, error_count, total_removed_missing, total_removed_geographic, error_files

def generate_summary_report(input_dir, output_file=None):
    """
    Generate a summary report of missing data and geographic outliers across all CSV files.
    
    Args:
        input_dir (str): Path to the input directory containing CSV files
        output_file (str): Path to save the summary report (optional)
    """
    input_path = Path(input_dir)
    csv_files = list(input_path.rglob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Generating summary report for {len(csv_files)} CSV files...")
    
    summary_data = []
    
    for csv_file in tqdm(csv_files, desc="Analyzing files"):
        try:
            stats = analyze_missing_data(csv_file)
            relative_path = csv_file.relative_to(input_path)
            
            summary_data.append({
                'file': str(relative_path),
                'total_rows': stats['total_rows'],
                'missing_bus_no': stats['missing_data'].get('bus_no', {}).get('count', 0),
                'missing_boarding_point': stats['missing_data'].get('boarding_point_name', {}).get('count', 0),
                'missing_pincode': stats['missing_data'].get('pincode', {}).get('count', 0),
                'missing_latitude': stats['missing_data'].get('latitude', {}).get('count', 0),
                'missing_longitude': stats['missing_data'].get('longitude', {}).get('count', 0),
                'missing_address': stats['missing_data'].get('address', {}).get('count', 0),
                'geographic_outliers': stats['geographic_outliers']
            })
            
        except Exception as e:
            print(f"Error analyzing {csv_file}: {str(e)}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate totals
    total_rows = summary_df['total_rows'].sum()
    total_missing_bus_no = summary_df['missing_bus_no'].sum()
    total_missing_boarding_point = summary_df['missing_boarding_point'].sum()
    total_missing_pincode = summary_df['missing_pincode'].sum()
    total_missing_latitude = summary_df['missing_latitude'].sum()
    total_missing_longitude = summary_df['missing_longitude'].sum()
    total_missing_address = summary_df['missing_address'].sum()
    total_geographic_outliers = summary_df['geographic_outliers'].sum()
    
    print("\n" + "="*80)
    print("ENHANCED SUMMARY REPORT")
    print("="*80)
    print(f"Total files analyzed: {len(summary_df)}")
    print(f"Total rows: {total_rows:,}")
    print(f"Files with missing data: {(summary_df.iloc[:, 2:8] != 0).any(axis=1).sum()}")
    print(f"Files with geographic outliers: {(summary_df['geographic_outliers'] != 0).sum()}")
    print()
    print("Missing data by column:")
    print(f"  Bus Number: {total_missing_bus_no:,} ({total_missing_bus_no/total_rows*100:.1f}%)")
    print(f"  Boarding Point: {total_missing_boarding_point:,} ({total_missing_boarding_point/total_rows*100:.1f}%)")
    print(f"  Pincode: {total_missing_pincode:,} ({total_missing_pincode/total_rows*100:.1f}%)")
    print(f"  Latitude: {total_missing_latitude:,} ({total_missing_latitude/total_rows*100:.1f}%)")
    print(f"  Longitude: {total_missing_longitude:,} ({total_missing_longitude/total_rows*100:.1f}%)")
    print(f"  Address: {total_missing_address:,} ({total_missing_address/total_rows*100:.1f}%)")
    print(f"  Geographic Outliers: {total_geographic_outliers:,} ({total_geographic_outliers/total_rows*100:.1f}%)")
    
    if output_file:
        summary_df.to_csv(output_file, index=False)
        print(f"\nDetailed report saved to: {output_file}")
    
    return summary_df

def main():
    """Main function to run the enhanced data cleaning process."""
    # Get the current directory
    current_dir = os.getcwd()
    
    # Define the input directory (Routes_Data)
    routes_data_dir = os.path.join(current_dir, "Routes_Data")
    
    # Check if Routes_Data directory exists
    if not os.path.exists(routes_data_dir):
        print(f"Error: Routes_Data directory not found at {routes_data_dir}")
        print("Please run this script from the directory containing Routes_Data")
        return
    
    print("Enhanced CSV Data Cleaning Tool")
    print("=" * 60)
    print("1. Generate summary report (includes geographic analysis)")
    print("2. Clean data by removing rows with missing bus information")
    print("3. Clean data by marking missing data")
    print("4. Clean data by filling with default values")
    print("5. Clean data with geographic filtering (remove outliers)")
    print("6. Comprehensive cleaning (remove missing data + geographic outliers)")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == '1':
        # Generate summary report
        generate_summary_report(routes_data_dir, "enhanced_missing_data_summary.csv")
        
    elif choice == '2':
        # Remove rows with missing bus information
        print("\nRemoving rows with missing bus information...")
        clean_csv_data(routes_data_dir, strategy='remove', filter_geographic=False)
        
    elif choice == '3':
        # Mark missing data
        print("\nMarking rows with missing bus information...")
        clean_csv_data(routes_data_dir, strategy='keep', filter_geographic=False)
        
    elif choice == '4':
        # Fill with default values
        print("\nFilling missing data with default values...")
        clean_csv_data(routes_data_dir, strategy='fill_default', filter_geographic=False)
        
    elif choice == '5':
        # Geographic filtering only
        print("\nRemoving geographic outliers only...")
        clean_csv_data(routes_data_dir, strategy='keep', filter_geographic=True)
        
    elif choice == '6':
        # Comprehensive cleaning
        print("\nPerforming comprehensive cleaning (missing data + geographic outliers)...")
        clean_csv_data(routes_data_dir, strategy='remove', filter_geographic=True)
        
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()