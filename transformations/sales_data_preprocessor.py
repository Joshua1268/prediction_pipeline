import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL 
from scipy.stats import zscore  
from typing import Tuple


class SalesDataPreprocessor:
    def __init__(self, seasonal_period: int = 7):
        self.seasonal_period = seasonal_period


    def _filter_by_recurrence(self, df: pd.DataFrame) -> pd.DataFrame:
        recurrence = (
            df.groupby("standard_name")["count_orderId"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        recurrence["segment"] = pd.qcut(
            recurrence["count_orderId"], q=2, labels=["Faible", "Eleve"]
        )

        class_A = recurrence[
            recurrence["segment"] == "Eleve"
        ]["standard_name"].tolist()
        filtered = df[df["standard_name"].isin(class_A)]

        return filtered.sort_values("delivery_date")


    def _filter_by_min_sales_days(self, df: pd.DataFrame, min_days: int) -> pd.DataFrame:

        has_orders = df[df["count_orderId"] > 0]
        days_with_sales = has_orders.groupby("standard_id")[
            "delivery_date"
        ].nunique()

        valid_standard_ids = days_with_sales[days_with_sales >= min_days].index
        filtered_data = df[df["standard_id"].isin(valid_standard_ids)].copy()

        ids_all = set(df["standard_id"].unique())
        ids_valid = set(filtered_data["standard_id"].unique())
        to_exclude = ids_all - ids_valid

        if to_exclude:
            excluded_names = (
                df[df["standard_id"].isin(to_exclude)]
                .groupby("standard_name", as_index=False)["count_orderId"]
                .sum()
            )
            excluded_names["segment"] = "Faible"
            exclude_list = excluded_names["standard_name"].unique()
            filtered_data = filtered_data[~filtered_data["standard_name"].isin(
                exclude_list
                )]

        return filtered_data


    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["quantity"] = df["quantity"].clip(lower=0).astype(float)
        df["standard_name"] = df["standard_name"].str.strip()
        df["is_outlier"] = False

        grouped = df.groupby("standard_name")
        for name, group in grouped:
            if len(group) < 10:
                continue

            ts = group.set_index("delivery_date")["quantity"].copy()
            resid = ts.values

            if len(ts) >= 2 * self.seasonal_period and ts.nunique() > 1:
                try:
                    ts = ts.asfreq("D", fill_value=0)
                    stl = STL(
                        ts,
                        period=self.seasonal_period,
                        robust=True
                        ).fit()
                    resid_series = stl.resid.reindex(ts.index, fill_value=0)
                    if len(resid_series) != len(group):
                        continue
                    resid = resid_series.values
                except Exception:
                    pass

            q1, q3 = np.percentile(resid, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - 2.5 * iqr, q3 + 2.5 * iqr

            z_scores = pd.Series(0.0, index=group.index)
            valid_sales = group["quantity"][group["quantity"] > 0]
            if len(valid_sales) > 1 and valid_sales.std() > 0:
                z = zscore(valid_sales)
                z_scores.loc[valid_sales.index] = z

            outlier_mask = (
                ((group["quantity"] > 0) & ((resid < lower) | (resid > upper)))
                | ((group["quantity"] > 0) & (np.abs(z_scores) > 2.0))
            )
            df.loc[group.index, "is_outlier"] = outlier_mask

        df["quantity_cleaned"] = df["quantity"]

        for name, group in df.groupby("standard_name"):
            ts = group.set_index("delivery_date")["quantity_cleaned"].copy()
            mask = group.set_index("delivery_date")["is_outlier"]
            ts[mask] = np.nan
            ts = ts.interpolate(method="time", limit_direction="both")

            if ts.isna().any():
                median = group.loc[
                    (~group["is_outlier"]) & (group["quantity"] > 0),
                    "quantity"
                    ].median()
                ts = ts.fillna(median if not pd.isna(median) else 0)

            df.loc[group.index, "quantity_cleaned"] = ts.values

        df["quantity"] = df["quantity_cleaned"]
        return df.drop(
            columns=["is_outlier", "quantity_cleaned"],
            errors="ignore"
            )


    def _split_train_test(self, df: pd.DataFrame, test_days: int = 60
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        last_date = df["delivery_date"].max()
        split_date = last_date - pd.Timedelta(days=test_days)

        train = df[df["delivery_date"] < split_date].copy()
        test = df[df["delivery_date"] >= split_date].copy()

        return train, test
    

    def _add_lag(self, df, col, n):
        df = df.copy()
        lag_col = f"{col}_lag{n}"
        
        if col == 'active':
            ## Les status des clients sont définis par semaine
            ## Donc on décale par semaine (7 jours)
            ## et on remplit les valeurs manquantes par propagation
            df_source = df[['delivery_date', 'active']].drop_duplicates(subset='delivery_date').copy()
            
            df_source['delivery_date'] = df_source['delivery_date'] + pd.to_timedelta(7 * n, unit='d')
            
            df_source.rename(columns={'active': lag_col}, inplace=True)
            
            df = pd.merge(df, df_source, on='delivery_date', how='left')
            
            df = df.sort_values(by='delivery_date')
            df[lag_col] = df[lag_col].ffill().fillna(0).astype(int)
        else:
            df[lag_col] = df[col].shift(n)
        
        return df
    

    def fillna_columns(self, columns: list, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(by=["standard_id", "delivery_date"])
        
        for col in columns:
            if col == "unit_price":
                df[col] = df[col].astype('float64')
            
                df[col] = df[col].replace(0, np.nan)
                
                df[col] = df.groupby('standard_id')[col].transform(
                    lambda x: x.fillna(x.median())
                )
            else:
                df[col] = df[col].fillna(0)
        return df