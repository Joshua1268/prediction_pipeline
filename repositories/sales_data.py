from datetime import datetime
import pandas as pd  
import holidays
from shared.data_loader import DatabaseConnector
from typing import Optional, List


class SalesRepository:
    def __init__(self):
        self.connector = DatabaseConnector()
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        try:
            self.conn = self.connector.connect()
            if self.conn is None:
                raise ConnectionError("Impossible d'établir la connexion à la base de données.")
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation de la connexion à la base de données: {e}")
            self.conn = None # S'assurer que self.conn est None en cas d'échec
            raise

    def get_sales_data(self) -> pd.DataFrame:
        query = """WITH RECURSIVE date_series AS (
                        SELECT DATE('2024-05-20') AS delivery_date
                    UNION ALL
                        SELECT delivery_date + INTERVAL 1 DAY
                        FROM date_series
                        WHERE delivery_date < %s
                    ),
        product_standards_data AS (
            SELECT DISTINCT
                ps.id            AS standard_id,
                ps.name          AS standard_name,
                ut2.title        AS default_unit
            FROM product_standards ps
            JOIN products p ON ps.id = p.product_standard_id
            JOIN unit_translations ut2 ON ut2.unit_id = ps.unit_id
            WHERE ps.deleted_at IS NULL
            AND p.deleted_at IS NULL
            AND ut2.deleted_at IS NULL
            AND p.category_id NOT IN (8, 10, 11)
        ),
        all_combinations AS (
            SELECT
                ds.delivery_date,
                psd.standard_id,
                psd.standard_name,
                psd.default_unit
            FROM date_series ds
            CROSS JOIN product_standards_data psd
        ),
        original_data AS (
            SELECT
                o.delivery_date,
                COUNT(DISTINCT o.id)  AS count_orderId,
                p.product_standard_id       AS standard_id,
                ps.name                     AS standard_name,
                SUM(od.quantity * p.weight) AS quantity,
                AVG(od.unit_price/p.weight) AS unit_price ,
                ut.title                   AS default_unit
            FROM orders o
            JOIN order_details od  ON o.id = od.order_id
            JOIN stocks st         ON st.id = od.stock_id
            JOIN products p        ON p.id = st.countable_id
            JOIN product_standards ps ON ps.id = p.product_standard_id
            JOIN categories c      ON c.id = p.category_id
            JOIN unit_translations ut ON ut.unit_id = ps.unit_id
            WHERE o.status  NOT IN ('canceled','proforma')
            AND o.delivery_date <= %s
            AND od.deleted_at  IS NULL
            AND p.deleted_at   IS NULL
            AND ps.deleted_at  IS NULL
            AND ut.deleted_at  IS NULL
            AND o.shop_id      <> 43
            AND p.category_id  NOT IN (8, 10, 11)
            GROUP BY o.delivery_date, p.product_standard_id, ps.name
        )
        SELECT
            ac.delivery_date,
            DAY(ac.delivery_date) AS day,
            WEEKDAY(ac.delivery_date) +1 AS day_of_week,
            WEEK(ac.delivery_date, 1) AS week,
            FLOOR((DAY(ac.delivery_date) - 1) / 7) + 1 AS week_of_month,
            MONTH(ac.delivery_date) AS month_of_year,
            ac.standard_id,
            ac.standard_name,
            COALESCE(od.count_orderId, 0) AS count_orderId,
            COALESCE(od.quantity, 0)      AS quantity,
            COALESCE(od.unit_price, 0)   AS unit_price ,
            ac.default_unit
        FROM all_combinations ac
        LEFT JOIN original_data od
        ON ac.delivery_date = od.delivery_date
        AND ac.standard_id   = od.standard_id
        ORDER BY ac.delivery_date ASC;"""
        
        if self.conn is None:
            raise RuntimeError("Connexion à la base de données non établie.")
            
        try:
            df = pd.read_sql_query(
                query, self.conn, params=(self.end_date, self.end_date)
            )
            return df
        except Exception as e:
            print(f"❌ Erreur lors de l'exécution de get_sales_data: {e}")
            
            return pd.DataFrame() 

    def get_stockouts(self) -> pd.DataFrame:
        query = """SELECT
                o.delivery_date,
                ps.id AS standard_id,
                COUNT(DISTINCT od.order_id) AS stockout
            FROM orders o
            JOIN order_details od ON o.id = od.order_id
            JOIN stocks s ON s.id = od.stock_id
            JOIN products p ON p.id = s.countable_id
            JOIN product_standards ps ON ps.id = p.product_standard_id
            WHERE od.reason_delete ='rupture'
            GROUP BY ps.name,o.delivery_date"""
       
        if self.conn is None:
            raise RuntimeError("Connexion à la base de données non établie.")
            
        try:
            df = pd.read_sql(query, self.conn)
            return df
        except Exception as e:
            print(f"❌ Erreur lors de l'exécution de get_stockouts: {e}")
            return pd.DataFrame()

    def get_customer_status(self) -> pd.DataFrame:
        query = """WITH RECURSIVE weeks AS (
                SELECT 
                    DATE_ADD('2023-01-02', INTERVAL (week_num * 7) DAY) AS week_start,
                    week_num + 1 AS week_number
                FROM (
                    SELECT @row := @row + 1 AS week_num
                    FROM (
                        SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
                        UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9
                        UNION ALL SELECT 10 UNION ALL SELECT 11
                    ) a,
                    (
                        SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
                        UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9
                        UNION ALL SELECT 10 UNION ALL SELECT 11
                    ) b,
                    (SELECT @row := 0) r
                    LIMIT 208  
                ) week_nums
            ), 
            shops AS (
                SELECT DISTINCT id AS shop_id, created_at, open, updated_at
                FROM backend.shops
            ), 
            shop_orders_by_week AS (
                SELECT
                    bo.shop_id,
                    DATE_FORMAT(bo.delivery_date, '%Y-%m-%d') AS delivery_date,
                    CEIL(DATEDIFF(DATE_FORMAT(bo.delivery_date, '%Y-%m-%d'), '2023-01-01') / 7) AS week_number
                FROM backend.orders bo
                WHERE bo.status NOT IN ('canceled','proforma')
            ), 
            shop_first_order_week AS (
                SELECT
                    s.id AS shop_id,
                    MIN(CEIL(DATEDIFF(DATE_FORMAT(bo.delivery_date, '%Y-%m-%d'), '2023-01-01') / 7)) AS first_order_week
                FROM backend.orders bo
                JOIN backend.shops s ON bo.shop_id = s.id
                WHERE bo.status NOT IN ('canceled','proforma')
                GROUP BY s.id
            ), 
            shop_activity AS (
                SELECT 
                    s.id AS shop_id,
                    w.week_start,
                    w.week_number,
                    IFNULL(COUNT(so.delivery_date), 0) AS order_count,
                    COALESCE((SELECT first_order_week FROM shop_first_order_week WHERE shop_first_order_week.shop_id = s.id), NULL) AS first_order_week
                FROM weeks w
                JOIN backend.shops s ON s.created_at < DATE_ADD(w.week_start, INTERVAL 7 DAY)
                LEFT JOIN shop_orders_by_week so ON w.week_number = so.week_number AND s.id = so.shop_id
                WHERE w.week_start <= CURRENT_DATE() AND s.status = 'approved' AND (s.open = 1 OR (s.open = 0 AND w.week_start < DATE_FORMAT(s.updated_at, '%Y-%m-%d')))
                GROUP BY s.id, w.week_start, w.week_number
            ), 
            shop_status AS (
                SELECT 
                    sa.shop_id,
                    sa.week_start,
                    sa.week_number,
                    sa.order_count,
                    COALESCE(
                        CASE
                            WHEN sa.first_order_week IS NULL THEN 'prospect'
                            WHEN sa.week_number < sa.first_order_week THEN 'prospect'
                            WHEN sa.order_count > 0 AND sa.week_number = sa.first_order_week THEN 'new'
                            WHEN sa.order_count > 0 AND LAG(sa.order_count, 1) OVER (PARTITION BY sa.shop_id ORDER BY sa.week_start) > 0 THEN 'active'
                            WHEN sa.order_count = 0 AND LAG(sa.order_count, 1) OVER (PARTITION BY sa.shop_id ORDER BY sa.week_start) > 0 THEN 'churned'
                            WHEN sa.order_count > 0 AND LAG(sa.order_count, 1) OVER (PARTITION BY sa.shop_id ORDER BY sa.week_start) = 0 THEN 'resurrected'
                            WHEN sa.order_count = 0 AND sa.week_number >= sa.first_order_week THEN 'inactive'
                        END,
                        'prospect'
                    ) AS status
                FROM shop_activity sa
                ORDER BY sa.shop_id, sa.week_start
            )
            SELECT 
                ss.week_start, 
                ss.status,
                COUNT(DISTINCT ss.shop_id) AS shop_count
            FROM shop_status ss
            GROUP BY ss.week_start, ss.status
            ORDER BY ss.week_start ASC, FIELD(ss.status, 'prospect', 'new', 'active', 'inactive', 'churned', 'resurrected');
        """
        if self.conn is None:
            raise RuntimeError("Connexion à la base de données non établie.")
            
        try:
            df = pd.read_sql(query,  self.conn)
            return df
        except Exception as e:
            print(f"❌ Erreur lors de l'exécution de get_discounts: {e}")
            return pd.DataFrame()

    def get_discounts(self) -> pd.DataFrame:
        query = """
            SELECT
                o.delivery_date AS delivery_date,
                ps.id AS standard_id,
                ps.name AS standard_name,
                od.discount
            FROM order_details od
            JOIN orders o On o.id = od.order_id
            JOIN stocks s On s.id = od.stock_id
            JOIN products p ON p.id = s.countable_id
            JOIN product_standards ps ON ps.id = p.product_standard_id
            WHERE od.deleted_at IS NULL
                AND o.delivery_date <= %s
                AND o.status NOT IN ('canceled', 'proforma')
                AND ps.deleted_at IS NULL
            GROUP BY o.delivery_date,ps.name
            HAVING od.discount > 0
        """
        
        if self.conn is None:
            raise RuntimeError("Connexion à la base de données non établie.")
            
        try:
            df = pd.read_sql(query,  self.conn, params=(self.end_date,))
            return df
        except Exception as e:
            print(f"❌ Erreur lors de l'exécution de get_discounts: {e}")
            return pd.DataFrame()

    def get_holidays(
        self,
        start_date: str,
        end_date: str = datetime.today().strftime('%Y-%m-%d')
    ) -> pd.DataFrame:
        """Get holidays in Cote d'Ivoire between start_date and end_date."""
        
        try:
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)

            if start_date_dt > end_date_dt:
                raise ValueError("start_date doit être avant end_date")

            delivery_date = pd.date_range(start=start_date_dt, end=end_date_dt)

            days_data = pd.DataFrame({'delivery_date': delivery_date})
            days_data['delivery_date'] = days_data['delivery_date'].dt.strftime(
                '%Y-%m-%d'
            )

            days_data['date_dt'] = pd.to_datetime(days_data['delivery_date'])

            ci_holidays = holidays.CountryHoliday(
                'CI',
                years=range(
                    2024,
                    datetime.today().year + 1
                )
            )

            days_data['is_holiday'] = days_data['date_dt'].isin(
                ci_holidays
                ).astype(int)

            days_data.drop(columns=['date_dt'], inplace=True)
            return days_data
            
        except ValueError as ve:
            print(f"❌ Erreur de validation des dates dans get_holidays: {ve}")
            raise # Relance l'erreur de validation
        except Exception as e:
            print(f"❌ Erreur lors de l'exécution de get_holidays: {e}")
            return pd.DataFrame()
        
    
   