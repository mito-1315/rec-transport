#!/usr/bin/env python3
"""
Script to convert all xlsx files in Routes_Data directory to CSV format.
Preserves the directory structure and creates corresponding CSV files.
"""

import os
import pandas as pd
from pathlib import Path
import glob
import sys
from tqdm import tqdm

def convert_xlsx_to_csv(input_dir, output_dir=None, verbose=True):
    """
    Convert all xlsx files in the input directory (and subdirectories) to CSV format.
    
    Args:
        input_dir (str): Path to the input directory containing xlsx files
        output_dir (str): Path to the output directory (optional, defaults to input_dir)
        verbose (bool): Whether to print detailed progress information
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Convert to Path objects for easier handling
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all xlsx files recursively
    xlsx_files = list(input_path.rglob("*.xlsx"))
    
    if not xlsx_files:
        print(f"No xlsx files found in {input_dir}")
        return
    
    if verbose:
        print(f"Found {len(xlsx_files)} xlsx files to convert:")
        for file in xlsx_files:
            print(f"  - {file}")
        print()
    
    converted_count = 0
    error_count = 0
    errors = []
    
    # Use tqdm for progress bar if verbose is True
    file_iterator = tqdm(xlsx_files, desc="Converting files") if verbose else xlsx_files
    
    for xlsx_file in file_iterator:
        try:
            # Read the Excel file
            if verbose:
                print(f"Converting: {xlsx_file}")
            df = pd.read_excel(xlsx_file)
            
            # Create the corresponding CSV path
            relative_path = xlsx_file.relative_to(input_path)
            csv_file = output_path / relative_path.with_suffix('.csv')
            
            # Create output directory if it doesn't exist
            csv_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as CSV
            df.to_csv(csv_file, index=False)
            if verbose:
                print(f"  ✓ Saved: {csv_file}")
            converted_count += 1
            
        except Exception as e:
            error_msg = f"Error converting {xlsx_file}: {str(e)}"
            if verbose:
                print(f"  ✗ {error_msg}")
            errors.append(error_msg)
            error_count += 1
    
    # Print summary
    print(f"\nConversion complete!")
    print(f"Successfully converted: {converted_count} files")
    if error_count > 0:
        print(f"Errors: {error_count} files")
        if verbose:
            print("\nError details:")
            for error in errors:
                print(f"  - {error}")
    
    return converted_count, error_count, errors

def main():
    """Main function to run the conversion."""
    # Get the current directory
    current_dir = os.getcwd()
    
    # Define the input directory (Routes_Data)
    routes_data_dir = os.path.join(current_dir, "Routes_Data")
    
    # Check if Routes_Data directory exists
    if not os.path.exists(routes_data_dir):
        print(f"Error: Routes_Data directory not found at {routes_data_dir}")
        print("Please run this script from the directory containing Routes_Data")
        print(f"Current directory: {current_dir}")
        return
    
    print(f"Starting conversion of xlsx files in: {routes_data_dir}")
    print("=" * 60)
    
    # Convert all xlsx files
    converted, errors, error_list = convert_xlsx_to_csv(routes_data_dir)
    
    print("=" * 60)
    if errors == 0:
        print("✅ All files converted successfully!")
    else:
        print(f"⚠️  Conversion completed with {errors} errors.")
    
    print("Check the Routes_Data directory for the converted CSV files.")

if __name__ == "__main__":
    main() 