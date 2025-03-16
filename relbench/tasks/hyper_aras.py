import duckdb
import pandas as pd
from relbench.datasets import register_task
from relbench.base import Database, EntityTask, RecommendationTask, Table, TaskType
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
    mae,
    r2,
    rmse,
    roc_auc,
)


# Task 1: Predict articles each customer will purchase in the next 7 days
class CustomerArticlePurchaseTask(RecommendationTask):
    r"""Predict the list of articles each customer will purchase in the next seven days."""
    
    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "customers_id"
    src_entity_table = "customer"
    dst_entity_col = "articles_id"
    dst_entity_table = "article"
    time_col = "d_dat"
    timedelta = pd.Timedelta(days=7)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                transactions.customers_id,
                LIST(DISTINCT transactions.articles_id) AS articles_id
            FROM
                timestamp_df t
            LEFT JOIN
                transactions
            ON
                transactions.datetime > t.timestamp AND
                transactions.datetime <= t.timestamp + INTERVAL '{self.timedelta.days} days'
            GROUP BY
                t.timestamp,
                transactions.customers_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )


# Task 2: Predict customer churn (no purchases in the next week)
class CustomerChurnTask(EntityTask):
    r"""Predict whether a customer will churn (no transactions in the next 7 days)."""
    
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "customers_id"
    entity_table = "customer"
    time_col = "d_dat"
    target_col = "churn"
    timedelta = pd.Timedelta(days=7)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        transactions = db.table_dict["transactions"].df
        customers = db.table_dict["customer"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                c.customers_id,
                CAST(
                    NOT EXISTS (
                        SELECT 1
                        FROM transactions
                        WHERE
                            transactions.customers_id = c.customers_id AND
                            transactions.datetime > t.timestamp AND
                            transactions.datetime <= t.timestamp + INTERVAL '{self.timedelta.days} days'
                    ) AS INTEGER
                ) AS churn
            FROM
                timestamp_df t,
                customer c
            WHERE
                EXISTS (
                    SELECT 1
                    FROM transactions
                    WHERE
                        transactions.customers_id = c.customers_id AND
                        transactions.datetime > t.timestamp - INTERVAL '{self.timedelta.days} days' AND
                        transactions.datetime <= t.timestamp
                )
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


# Task 3: Predict article sales in the next 7 days
class ArticleSalesTask(EntityTask):
    r"""Predict the total sales for an article (sum of `price_purchase`) in the next 7 days."""
    
    task_type = TaskType.REGRESSION
    entity_col = "articles_id"
    entity_table = "article"
    time_col = "d_dat"
    target_col = "sales"
    timedelta = pd.Timedelta(days=7)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        transactions = db.table_dict["transactions"].df
        articles = db.table_dict["article"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                a.articles_id,
                COALESCE(SUM(transactions.price_purchase), 0) AS sales
            FROM
                timestamp_df t,
                article a
            LEFT JOIN
                transactions
            ON
                transactions.articles_id = a.articles_id AND
                transactions.datetime > t.timestamp AND
                transactions.datetime <= t.timestamp + INTERVAL '{self.timedelta.days} days'
            GROUP BY
                t.timestamp,
                a.articles_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

