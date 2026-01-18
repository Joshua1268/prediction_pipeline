#!/usr/bin/env python3

import csv
import random
from datetime import datetime, timedelta
import os
from typing import List, Tuple


class FruitsVegetablesDataGenerator:
    def __init__(self, start_date: str = "2023-01-01", end_date: str = "2024-12-31"):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.dates = []
        current_date = self.start_date
        while current_date <= self.end_date:
            self.dates.append(current_date)
            current_date += timedelta(days=1)

        # Fruit and vegetable categories
        self.categories = [
            'Organic Fruits', 'Organic Vegetables', 'Exotic Fruits', 'Citrus Fruits',
            'Stone Fruits', 'Berries', 'Leafy Greens', 'Root Vegetables',
            'Cruciferous Vegetables', 'Allium Vegetables', 'Summer Vegetables', 'Winter Vegetables'
        ]

        self.fruit_names = [
            'Apples', 'Bananas', 'Oranges', 'Strawberries', 'Grapes', 'Pineapples',
            'Mangoes', 'Blueberries', 'Raspberries', 'Blackberries', 'Peaches',
            'Plums', 'Cherries', 'Pears', 'Kiwi', 'Avocados', 'Lemons', 'Limes',
            'Watermelon', 'Cantaloupe', 'Honeydew Melon', 'Pomegranate', 'Dragon Fruit'
        ]

        self.vegetable_names = [
            'Spinach', 'Lettuce', 'Kale', 'Arugula', 'Broccoli', 'Cauliflower',
            'Brussels Sprouts', 'Cabbage', 'Carrots', 'Radishes', 'Beets', 'Turnips',
            'Onions', 'Garlic', 'Leeks', 'Scallions', 'Tomatoes', 'Bell Peppers',
            'Zucchini', 'Cucumber', 'Eggplant', 'Asparagus', 'Green Beans', 'Peas',
            'Corn', 'Potatoes', 'Sweet Potatoes', 'Pumpkin', 'Butternut Squash'
        ]

    def generate_products(self, num_fruits: int = 20, num_vegetables: int = 25) -> List[dict]:
        """Generate synthetic fruit and vegetable product data"""
        products = []

        # Generate fruit products
        for i in range(num_fruits):
            fruit = random.choice(self.fruit_names)
            price = round(random.uniform(2.0, 15.0), 2)  # Generate price first
            cost = round(random.uniform(1.0, price * 0.7), 2)  # Then calculate cost based on price
            fruit_product = {
                'product_id': f'FRUIT{i+1:03d}',
                'product_name': f'{fruit}',
                'category': random.choice(['Organic Fruits', 'Exotic Fruits', 'Citrus Fruits', 'Stone Fruits', 'Berries']),
                'price': price,  # Reasonable prices for fruits
                'cost': cost,  # Cost is typically 60-70% of price
                'weight_kg': round(random.uniform(0.1, 2.0), 2),  # Weight in kg
                'seasonal_factor': random.uniform(0.5, 1.8),  # Seasonal demand factor
                'trend_factor': random.uniform(0.8, 1.3),     # Trend factor
                'shelf_life_days': random.randint(3, 14)      # Perishability factor
            }
            products.append(fruit_product)

        # Generate vegetable products
        for i in range(num_vegetables):
            veg = random.choice(self.vegetable_names)
            price = round(random.uniform(1.5, 12.0), 2)  # Generate price first
            cost = round(random.uniform(0.8, price * 0.7), 2)  # Then calculate cost based on price
            veg_product = {
                'product_id': f'VEGETABLE{i+1:03d}',
                'product_name': f'{veg}',
                'category': random.choice([
                    'Leafy Greens', 'Root Vegetables', 'Cruciferous Vegetables',
                    'Allium Vegetables', 'Summer Vegetables', 'Winter Vegetables', 'Organic Vegetables'
                ]),
                'price': price,  # Reasonable prices for vegetables
                'cost': cost,  # Cost is typically 60-70% of price
                'weight_kg': round(random.uniform(0.1, 3.0), 2),  # Weight in kg
                'seasonal_factor': random.uniform(0.5, 1.8),  # Seasonal demand factor
                'trend_factor': random.uniform(0.8, 1.3),     # Trend factor
                'shelf_life_days': random.randint(3, 21)      # Perishability factor
            }
            products.append(veg_product)

        return products

    def generate_sales_data(self, products: List[dict]) -> List[dict]:
        """Generate synthetic sales data with seasonal patterns for fruits and vegetables"""
        sales_data = []

        for product in products:
            for date in self.dates:
                # Base demand calculation with seasonal and trend factors
                base_demand = max(0, random.gauss(30, 15))  # Base demand with some randomness

                # Apply seasonal factor based on product type and season
                month_factor = 1.0

                # Seasonal adjustments for fruits
                if 'fruit' in product['category'].lower():
                    if date.month in [6, 7, 8]:  # Summer - higher demand for many fruits
                        month_factor = 1.4
                    elif date.month in [11, 12, 1]:  # Winter - citrus season
                        month_factor = 1.2
                    elif date.month in [3, 4, 5]:  # Spring - berry season
                        month_factor = 1.3

                # Seasonal adjustments for vegetables
                elif 'vegetable' in product['category'].lower():
                    if date.month in [6, 7, 8]:  # Summer - high demand for summer vegetables
                        month_factor = 1.5
                    elif date.month in [9, 10]:  # Fall - harvest season
                        month_factor = 1.3
                    elif date.month in [11, 12, 1, 2]:  # Winter - root vegetables in demand
                        month_factor = 1.1

                # Apply weekend effect
                day_factor = 1.0
                if date.weekday() >= 5:  # Weekend
                    day_factor = 1.15  # Slightly higher demand on weekends

                # Apply product-specific factors
                seasonal_effect = product['seasonal_factor']
                trend_effect = product['trend_factor']

                # Calculate final demand with noise
                demand = base_demand * month_factor * day_factor * seasonal_effect * trend_effect
                demand_with_noise = max(0, demand + random.gauss(0, demand * 0.15))

                # Convert demand to actual sales (considering perishability)
                actual_sales = max(0, int(demand_with_noise * random.uniform(0.6, 1.4)))

                # Calculate revenue and profit
                revenue = actual_sales * product['price']
                cost = actual_sales * product['cost']
                profit = revenue - cost

                sale_record = {
                    'date': date.strftime('%Y-%m-%d'),
                    'product_id': product['product_id'],
                    'product_name': product['product_name'],
                    'category': product['category'],
                    'quantity_sold': actual_sales,
                    'unit_price': product['price'],
                    'revenue': revenue,
                    'cost': cost,
                    'profit': profit,
                    'month': date.month,
                    'day_of_week': date.weekday(),
                    'is_weekend': 1 if date.weekday() >= 5 else 0,
                    'season': self.get_season(date.month),
                    'shelf_life_days': product['shelf_life_days']
                }

                sales_data.append(sale_record)

        return sales_data

    def get_season(self, month: int) -> str:
        """Helper method to determine season based on month"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    def generate_inventory_data(self, products: List[dict]) -> List[dict]:
        """Generate synthetic inventory data for perishable goods"""
        inventory_data = []

        for product in products:
            # Random initial stock level based on product type
            if 'fruit' in product['product_name'].lower():
                initial_stock = random.randint(20, 100)  # Lower for highly perishable fruits
            else:
                initial_stock = random.randint(30, 150)  # Higher for vegetables with longer shelf life

            for date in self.dates:
                # Simulate inventory changes considering perishability
                stock_change = random.randint(-20, 30)  # Restocking or usage
                current_stock = max(0, initial_stock + stock_change)

                inventory_record = {
                    'date': date.strftime('%Y-%m-%d'),
                    'product_id': product['product_id'],
                    'current_stock': current_stock,
                    'reorder_level': random.randint(5, 20),  # Lower reorder levels for perishables
                    'reorder_quantity': random.randint(30, 80),
                    'lead_time_days': random.randint(1, 5),  # Faster lead times for perishables
                    'shelf_life_days': product['shelf_life_days']
                }

                inventory_data.append(inventory_record)

        return inventory_data

    def generate_customer_behavior(self, products: List[dict]) -> List[dict]:
        """Generate synthetic customer behavior data for fruits and vegetables"""
        customer_data = []

        for product in products:
            for date in self.dates:
                # Customer engagement metrics specific to fruits and vegetables
                customer_record = {
                    'date': date.strftime('%Y-%m-%d'),
                    'product_id': product['product_id'],
                    'page_views': max(0, int(random.gauss(80, 25))),
                    'add_to_cart': max(0, int(random.gauss(15, 8))),
                    'purchases': max(0, int(random.gauss(8, 4))),
                    'conversion_rate': random.uniform(0.08, 0.25),
                    'avg_rating': round(random.uniform(3.5, 4.8), 1),  # Higher ratings for fresh produce
                    'reviews_count': max(0, int(random.gauss(3, 2))),
                    'organic_preference': 1 if 'Organic' in product['category'] else 0
                }

                customer_data.append(customer_record)

        return customer_data

    def generate_all_data(self) -> Tuple[List[dict], List[dict], List[dict]]:
        """Generate all synthetic datasets for fruits and vegetables"""
        print("Generating fruit and vegetable products...")
        products = self.generate_products(num_fruits=20, num_vegetables=25)

        print("Generating synthetic sales data...")
        sales_data = self.generate_sales_data(products)

        print("Generating synthetic inventory data...")
        inventory_data = self.generate_inventory_data(products)

        print("Generating synthetic customer behavior data...")
        customer_data = self.generate_customer_behavior(products)

        return sales_data, inventory_data, customer_data


def save_datasets_as_csv(sales_data: List[dict], inventory_data: List[dict], customer_data: List[dict], output_dir: str = "data"):
    """Save generated datasets to CSV files"""
    os.makedirs(output_dir, exist_ok=True)

    # Save sales data
    sales_file_path = os.path.join(output_dir, "fv_sales_data.csv")
    with open(sales_file_path, 'w', newline='') as csvfile:
        if sales_data:
            fieldnames = sales_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sales_data)

    # Save inventory data
    inventory_file_path = os.path.join(output_dir, "fv_inventory_data.csv")
    with open(inventory_file_path, 'w', newline='') as csvfile:
        if inventory_data:
            fieldnames = inventory_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(inventory_data)

    # Save customer data
    customer_file_path = os.path.join(output_dir, "fv_customer_data.csv")
    with open(customer_file_path, 'w', newline='') as csvfile:
        if customer_data:
            fieldnames = customer_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(customer_data)

    print(f"Fruits and vegetables datasets saved to {output_dir}/ directory")


if __name__ == "__main__":
    # Generate synthetic data for fruits and vegetables
    generator = FruitsVegetablesDataGenerator(start_date="2023-01-01", end_date="2024-12-31")
    sales_data, inventory_data, customer_data = generator.generate_all_data()

    # Save datasets
    save_datasets_as_csv(sales_data, inventory_data, customer_data)

    print("Fruits and vegetables synthetic data generation completed!")
    print(f"Sales data records: {len(sales_data)}")
    print(f"Inventory data records: {len(inventory_data)}")
    print(f"Customer data records: {len(customer_data)}")

    # Display sample of sales data
    print("\nSample of sales data:")
    if sales_data:
        print(sales_data[0])