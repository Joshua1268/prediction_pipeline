import pandas as pd  
from transformations.sales_data_preprocessor import SalesDataPreprocessor


class SalesDataBuilder (SalesDataPreprocessor):
    def __init__(self, df_discount, customer_status, holidays, stockout):
        self.df_discount = df_discount
        self.customer_status = customer_status
        self.holidays = holidays
        self.stockout = stockout

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self._add_discount_feature(data)
        data = self._add_customer_status(data)
        data = self._add_holidays(data)
        data = self._add_stockout(data)
        data = self.fillna_columns(['unit_price', 'stockout'], data)
        return data

    def _add_discount_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        
        df = self.df_discount[['delivery_date', 'standard_id']] \
                .assign(in_discount=1).drop_duplicates()
        return data.merge(
            df, on=['delivery_date', 'standard_id'], how='left'
            ).fillna({'in_discount': 0}).astype({'in_discount': 'int8'})

    def _add_customer_status(self, data: pd.DataFrame) -> pd.DataFrame:

        data['delivery_date'] = pd.to_datetime(data['delivery_date'])

        pivot_status = self.customer_status.pivot_table(
            index='week_start',
            columns='status',
            values='shop_count',
            fill_value=0
        ).reset_index()

        pivot_status['week_start'] = pd.to_datetime(pivot_status['week_start'])

        data['week_start'] = data['delivery_date'] - pd.to_timedelta(
            data['delivery_date'].dt.dayofweek, unit='D'
        )

        return pd.merge(data, pivot_status, on='week_start', how='left')

    def _add_holidays(self, data: pd.DataFrame) -> pd.DataFrame:

        self.holidays['delivery_date'] = pd.to_datetime(
            self.holidays['delivery_date'])
        return pd.merge(data, self.holidays, on='delivery_date', how='left')

    def _add_stockout(self, data: pd.DataFrame) -> pd.DataFrame:

        self.stockout['delivery_date'] = pd.to_datetime(
            self.stockout['delivery_date'])
        data = pd.merge(
            data,
            self.stockout,
            on=['delivery_date', 'standard_id'],
            how='left'
        )
        data['stockout'] = data['stockout'].fillna(0)
        return data