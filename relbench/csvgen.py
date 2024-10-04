import pandas as pd

# Load the transactional dataset into a pandas DataFrame
# Assuming you already have the dataset loaded from the image you provided
df = pd.read_csv('/media/danial/8034D28D34D28596/Projects/Hyper_data/Aras/Aras_transactions.csv')  # Replace with the actual file path

articles_df = df[['articles_id', 'art_name','Item Barcode','External Item Number', 'Department Name', 'group', 'Subgroup Name', 'group_id', 'whole_Branch_Name']].drop_duplicates()




# Step 2: Create the Customers table (unique customers with relevant columns)
customers_df = df[['customers_id', 'customers_no']].drop_duplicates()

# Step 3: Create the Branches table (unique branches with relevant columns)
branches_df = df[['BranchCode', 'BranchName']].drop_duplicates()

# Step 4: Create the Transactions table (linking all other tables with transactional details)
transactions_df = df[['factor_id', 'articles_id', 'customers_id', 'BranchCode', 'd_dat', 'salesTime',
                      'price_purchase', 'Return Amount', 'Discount_ratio', 'Quantity']]


# Step 2: Save the articles_df to a CSV file
articles_df.to_csv('Articles.csv', index=False)
customers_df.to_csv('Customers.csv', index=False)
branches_df.to_csv('Branches.csv', index=False)
transactions_df.to_csv('Transactions.csv', index=False)

print("CSV files generated successfully!")