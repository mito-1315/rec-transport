import pandas as pd
import os

# Path to your main CSV
master_csv = "data/Output Data/cleaned_student_locations_20250811_160344.csv"

# Output base folder
output_dir = "Routes_data_5800"

# Load the master CSV
df = pd.read_csv(master_csv)

# Days of the week you want to process
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# Time slots to match (based on how they appear in your data)
time_slots = {
    "8_am": "8:00 AM",
    "10_am": "10:00 AM",
    "3_pm": "3:00 PM",
    "5_pm": "5:00 PM",
    "Leave": "Leave"
}

# Columns to keep (day column will be added dynamically)
base_columns = [
    "id",           # ID
    "roll_no",      # Roll Number
    "email",        # Email
    "dept_name",    # Department
    "bus_no",
    "boarding_point_name",
    "pincode",
    "latitude",
    "longitude",
    "address"
]

for day in days:
    for folder_name, match_text in time_slots.items():
        # Filter rows for this time slot
        filtered = df[df[day].astype(str).str.contains(match_text, na=False)]

        if not filtered.empty:
            # Build columns list with the day's column inserted in right place
            columns_to_save = base_columns.copy()
            columns_to_save.insert(4, day)  # Insert day column after Department

            # Keep only required columns
            trimmed = filtered[columns_to_save]

            # Create folder
            save_path_dir = os.path.join(output_dir, day, folder_name)
            os.makedirs(save_path_dir, exist_ok=True)

            # Save to CSV
            save_path_file = os.path.join(save_path_dir, f"{folder_name}.csv")
            trimmed.to_csv(save_path_file, index=False)

print("✅ Trimmed CSVs split by day and time slot!")
