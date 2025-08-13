import pandas as pd

# Step 1: Read your CSV file
df = pd.read_csv(r"data/Output Data/cleaned_student_locations_20250811_160344.csv")

# Step 2: Filter rows where the day column ends with "5:00 PM"
filtered_mon = df[df["Monday"].astype(str).str.strip().str.endswith("5:00 PM")][["ID", "Roll Number", "Email", "Department", "Monday"]]
filtered_tue = df[df["Tuesday"].astype(str).str.strip().str.endswith("5:00 PM")][["ID", "Roll Number", "Email", "Department", "Tuesday"]]
filtered_wed = df[df["Wednesday"].astype(str).str.strip().str.endswith("5:00 PM")][["ID", "Roll Number", "Email", "Department", "Wednesday"]]
filtered_thur = df[df["Thursday"].astype(str).str.strip().str.endswith("5:00 PM")][["ID", "Roll Number", "Email", "Department", "Thursday"]]
filtered_fri = df[df["Friday"].astype(str).str.strip().str.endswith("5:00 PM")][["ID", "Roll Number", "Email", "Department", "Friday"]]
filtered_sat = df[df["Saturday"].astype(str).str.strip().str.endswith("5:00 PM")][["ID", "Roll Number", "Email", "Department", "Saturday"]]

# Step 3: Save filtered data to new CSV files
filtered_tue.to_csv(r"D:\Routes_Data\Routes_Data\Tuesday\5_pm\5_pm.csv", index=False)
filtered_wed.to_csv(r"D:\Routes_Data\Routes_Data\Wednesday\5_pm\5_pm.csv", index=False)
filtered_thur.to_csv(r"D:\Routes_Data\Routes_Data\Thursday\5_pm\5_pm.csv", index=False)
filtered_fri.to_csv(r"D:\Routes_Data\Routes_Data\Friday\5_pm\5_pm.csv", index=False)
filtered_sat.to_csv(r"D:\Routes_Data\Routes_Data\Saturday\5_pm\5_pm.csv", index=False)
filtered_mon.to_csv(r"D:\Routes_Data\Routes_Data\Monday\5_pm\5_pm.csv", index=False)

print("✅ Filtered CSVs saved")
