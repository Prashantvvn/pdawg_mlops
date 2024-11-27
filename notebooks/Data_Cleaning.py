import pandas as pd

# Load the dataset
file_path = r"C:\Users\bhara\mlops_project\multi_stock_data.csv"
raw_data = pd.read_csv(file_path)

# Use the first row as stock identifiers
stock_ids = raw_data.iloc[0, 1:]  # Skip the first column ('Price')
attributes = raw_data.columns[1:]  # Skip the 'Price' column header

# Generate new column names
new_columns = ['Date'] + [f"{stock}_{attr}" for stock, attr in zip(stock_ids, attributes)]

# Drop unnecessary rows and set the new header
cleaned_data = raw_data.iloc[2:].reset_index(drop=True)  # Keep data from row 2 onward
cleaned_data.columns = new_columns  # Rename columns

# Convert 'Date' column to datetime
cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'])

# Set 'Date' as the index
cleaned_data.set_index('Date', inplace=True)

# Display the cleaned data
print(cleaned_data.head())

# Save the cleaned dataset to a CSV file
output_file_path = r"C:\Users\bhara\mlops_project\cleaned_multi_stock_data.csv"
cleaned_data.to_csv(output_file_path)

print(f"Cleaned dataset saved to: {output_file_path}")
