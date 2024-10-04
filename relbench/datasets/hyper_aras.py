import os
import pandas as pd
import numpy as np
from relbench.base import Database, Dataset, Table

class TransactionalDataset(Dataset):
    # Set timestamps or other relevant information if needed
    val_timestamp = pd.Timestamp("2021-03-22")
    test_timestamp = pd.Timestamp("2021-03-29")

    def make_db(self) -> Database:
        # Path to your CSVs folder
        path = os.path.join("data", "transactional_data")
        customers = os.path.join(path, "Customers.csv")
        articles = os.path.join(path, "Articles.csv")
        branches = os.path.join(path, "Branches.csv")
        transactions = os.path.join(path, "Transactions.csv")

        # Ensure that CSV files exist in the specified path
        if not os.path.exists(customers):
            raise RuntimeError(f"Dataset not found at '{path}'. Please make sure the CSV files are in the correct folder.")

        # Read the CSV data into pandas DataFrames
        customers_df = pd.read_csv(customers)
        articles_df = pd.read_csv(articles)
        branches_df = pd.read_csv(branches)
        transactions_df = pd.read_csv(transactions)

        # Replace any missing or invalid values
        transactions_df = transactions_df.replace(r"^\\N$", np.nan, regex=True)

        # Combine date and time into a single 'datetime' column
        transactions_df['datetime'] = pd.to_datetime(transactions_df['d_dat'] + ' ' + transactions_df['salesTime'])

        # Convert other fields if necessary
        transactions_df['price_purchase'] = pd.to_numeric(transactions_df['price_purchase'], errors='coerce')
        transactions_df['Return Amount'] = pd.to_numeric(transactions_df['Return Amount'], errors='coerce')
        transactions_df['Discount_ratio'] = pd.to_numeric(transactions_df['Discount_ratio'], errors='coerce')
        transactions_df['Quantity'] = pd.to_numeric(transactions_df['Quantity'], errors='coerce')

        ################################################################################
        # Now we define the table structure and relationships.
        ################################################################################

        tables = {}

        # Articles table
        tables["articles"] = Table(
            df=pd.DataFrame(articles_df),
            fkey_col_to_pkey_table={},
            pkey_col="articles_id",
            time_col=None,
        )

        # Customers table
        tables["customers"] = Table(
            df=pd.DataFrame(customers_df),
            fkey_col_to_pkey_table={},
            pkey_col="customers_id",
            time_col=None,
        )

        # Branches table
        tables["branches"] = Table(
            df=pd.DataFrame(branches_df),
            fkey_col_to_pkey_table={},
            pkey_col="BranchCode",
            time_col=None,
        )

        # Transactions table
        tables["transactions"] = Table(
            df=pd.DataFrame(transactions_df),
            fkey_col_to_pkey_table={
                "articles_id": "articles",  # Foreign key to articles
                "customers_id": "customers",  # Foreign key to customers
                "BranchCode": "branches",  # Foreign key to branches
            },
            pkey_col="factor_id",
            time_col="datetime",  # Use the combined datetime column for time-based operations
        )

        return Database(tables)

# Usage
transactional_dataset = TransactionalDataset()
db = transactional_dataset.make_db()
