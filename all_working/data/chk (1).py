import pandas as pd

# Load your Excel and CSV
excel_mon = pd.read_excel(r"D:\Routes_Data\Routes_Data\Saturday\Leave\leave.xlsx")  
excel_mon1 = pd.read_excel(r"D:\Routes_Data\Routes_Data\Saturday\8_am\8_am.xlsx")
excel_mon2 = pd.read_excel(r"D:\Routes_Data\Routes_Data\Saturday\3_pm\3_pm.xlsx")
excel_mon3 = pd.read_excel(r"D:\Routes_Data\Routes_Data\Saturday\5_pm\5_pm.xlsx")
excel_mon4 = pd.read_excel(r"D:\Routes_Data\Routes_Data\Saturday\10_am\10_am.xlsx")
csv_df = pd.read_csv(r"D:\Routes_Data\Routes_Data\TransportMaster2025-08-07.csv")  

# Ensure ID columns are the same type
excel_mon1['ID'] = excel_mon1['ID'].astype(int)
excel_mon2['ID'] = excel_mon2['ID'].astype(int)
excel_mon3['ID'] = excel_mon3['ID'].astype(int)
excel_mon4['ID'] = excel_mon4['ID'].astype(int)
excel_mon['ID'] = excel_mon['ID'].astype(int)
csv_df['user'] = csv_df['user'].astype(int)

# Merge while keeping all Excel rows
merged_df = excel_mon.merge(csv_df, how="left", left_on="ID", right_on="user")
merged_df1 = excel_mon1.merge(csv_df, how="left", left_on="ID", right_on="user")
merged_df2 = excel_mon2.merge(csv_df, how="left", left_on="ID", right_on="user")
merged_df3 = excel_mon3.merge(csv_df, how="left", left_on="ID", right_on="user")
merged_df4 = excel_mon4.merge(csv_df, how="left", left_on="ID", right_on="user")

# Drop the duplicate 'user' column after merge
merged_df.drop(columns=['user'], inplace=True)
merged_df1.drop(columns=['user'], inplace=True)
merged_df2.drop(columns=['user'], inplace=True)
merged_df3.drop(columns=['user'], inplace=True)
merged_df4.drop(columns=['user'], inplace=True)


# Save result
merged_df.to_excel(r"D:\Routes_Data\Routes_Data\Saturday\Leave\leave.xlsx", index=False)
merged_df1.to_excel(r"D:\Routes_Data\Routes_Data\Saturday\8_am\8_am.xlsx", index=False)
merged_df2.to_excel(r"D:\Routes_Data\Routes_Data\Saturday\3_pm\3_pm.xlsx", index=False)
merged_df3.to_excel(r"D:\Routes_Data\Routes_Data\Saturday\5_pm\5_pm.xlsx", index=False)
merged_df4.to_excel(r"D:\Routes_Data\Routes_Data\Saturday\10_am\10_am.xlsx", index=False)


print("✅ Merged and saved successfully!")
