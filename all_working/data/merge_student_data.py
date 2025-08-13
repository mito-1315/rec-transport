#!/usr/bin/env python3
"""
Script to merge student_data.csv and student_flow_report.csv
Maps students based on roll number and email, combining all fields into a comprehensive dataset.
"""

import pandas as pd
import os
from datetime import datetime

def merge_student_data():
    """
    Merge student_data.csv and student_flow_report.csv based on roll number and email.
    Creates a comprehensive dataset with all fields from both files.
    """
    
    # File paths
    student_data_path = "data/student_data.csv"
    flow_report_path = "data/student_flow_report.csv"
    
    # Check if files exist
    if not os.path.exists(student_data_path):
        print(f"Error: {student_data_path} not found!")
        return
    
    if not os.path.exists(flow_report_path):
        print(f"Error: {flow_report_path} not found!")
        return
    
    print("Loading student data...")
    # Load student data
    student_df = pd.read_csv(student_data_path)
    print(f"Loaded {len(student_df)} records from student_data.csv")
    
    print("Loading flow report data...")
    # Load flow report data
    flow_df = pd.read_csv(flow_report_path)
    print(f"Loaded {len(flow_df)} records from student_flow_report.csv")
    
    # Clean and standardize column names for merging
    # Rename columns in flow_df to match student_df naming convention
    flow_df_renamed = flow_df.rename(columns={
        'Roll Number': 'roll_no',
        'Email': 'email',
        'Department': 'dept_name'
    })
    
    # Convert roll_no to string in both dataframes for consistent matching
    student_df['roll_no'] = student_df['roll_no'].astype(str)
    flow_df_renamed['roll_no'] = flow_df_renamed['roll_no'].astype(str)
    
    print("\nMerging datasets...")
    print(f"Student data columns: {list(student_df.columns)}")
    print(f"Flow report columns: {list(flow_df_renamed.columns)}")
    
    # Merge on roll_no and email (inner join to get only matching records)
    merged_df = pd.merge(
        student_df, 
        flow_df_renamed, 
        on=['roll_no', 'email'], 
        how='inner',
        suffixes=('', '_flow')
    )
    
    print(f"\nMerged {len(merged_df)} records successfully!")
    
    # Remove duplicate department columns if they exist
    if 'dept_name_flow' in merged_df.columns:
        # Keep the one from student_data.csv and drop the duplicate
        merged_df = merged_df.drop('dept_name_flow', axis=1)
        print("Removed duplicate department column")
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"merged_student_data_{timestamp}.csv"
    output_path = f"Output_data/{output_filename}"
    
    # Ensure output directory exists
    os.makedirs("Output_data", exist_ok=True)
    
    # Save merged data
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged data saved to: {output_path}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"- Total records in student_data.csv: {len(student_df)}")
    print(f"- Total records in student_flow_report.csv: {len(flow_df)}")
    print(f"- Successfully merged records: {len(merged_df)}")
    print(f"- Match rate: {(len(merged_df)/len(student_df)*100):.2f}%")
    
    # Show sample of merged data
    print(f"\nSample of merged data (first 5 rows):")
    print(merged_df.head())
    
    # Show all columns in final dataset
    print(f"\nFinal dataset columns ({len(merged_df.columns)} total):")
    for i, col in enumerate(merged_df.columns, 1):
        print(f"{i:2d}. {col}")
    
    return merged_df

def analyze_missing_matches():
    """
    Analyze which students couldn't be matched between the two datasets.
    """
    print("\n" + "="*50)
    print("ANALYZING MISSING MATCHES")
    print("="*50)
    
    # Load data again
    student_df = pd.read_csv("data/student_data.csv")
    flow_df = pd.read_csv("data/student_flow_report.csv")
    
    # Convert roll_no to string
    student_df['roll_no'] = student_df['roll_no'].astype(str)
    flow_df['Roll Number'] = flow_df['Roll Number'].astype(str)
    
    # Find students in student_data.csv but not in flow_report
    student_rolls = set(student_df['roll_no'])
    flow_rolls = set(flow_df['Roll Number'])
    
    missing_in_flow = student_rolls - flow_rolls
    missing_in_student = flow_rolls - student_rolls
    
    print(f"\nStudents in student_data.csv but missing from flow_report: {len(missing_in_flow)}")
    if missing_in_flow:
        print("Sample missing roll numbers:", list(missing_in_flow)[:10])
    
    print(f"\nStudents in flow_report but missing from student_data.csv: {len(missing_in_student)}")
    if missing_in_student:
        print("Sample missing roll numbers:", list(missing_in_student)[:10])
    
    # Check email matches
    student_emails = set(student_df['email'])
    flow_emails = set(flow_df['Email'])
    
    missing_emails_in_flow = student_emails - flow_emails
    missing_emails_in_student = flow_emails - student_emails
    
    print(f"\nEmails in student_data.csv but missing from flow_report: {len(missing_emails_in_flow)}")
    print(f"Emails in flow_report but missing from student_data.csv: {len(missing_emails_in_student)}")

if __name__ == "__main__":
    print("Student Data Merger")
    print("="*50)
    
    try:
        # Perform the merge
        merged_data = merge_student_data()
        
        # Analyze missing matches
        analyze_missing_matches()
        
        print("\n" + "="*50)
        print("MERGE COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"\nError during merge: {str(e)}")
        import traceback
        traceback.print_exc() 