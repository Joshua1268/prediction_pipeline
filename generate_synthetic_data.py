#!/usr/bin/env python3
"""
Synthetic Data Generator for Product Prediction System
This module generates realistic synthetic sales/product data for demonstration purposes.
"""
import csv
import random
from datetime import datetime, timedelta
import os
from typing import List, Tuple


class SimpleSyntheticDataGenerator:
    def __init__(self, start_date: str = "2023-01-01", end_date: str = "2024-12-31"):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.dates = []
        current_date = self.start_date
        while current_date <= self.end_date:
            self.dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Product categories and names
        self.categories = [
            'Electronics', 'Clothing', 'Home & Kitchen', 'Beauty', 'Sports', 
            'Books', 'Toys', 'Automotive', 'Health', 'Food'
        ]
        
        self.product_names = [
            'Smartphone', 'Laptop', 'Headphones', 'Watch', 'Tablet',
            'Shirt', 'Jeans', 'Dress', 'Shoes', 'Jacket',
            'Blender', 'Microwave', 'Coffee Maker', 'Toaster', 'Vacuum Cleaner',
            'Lipstick', 'Perfume', 'Shampoo', 'Conditioner', 'Face Cream',
            'Basketball', 'Yoga Mat', 'Dumbbells', 'Tennis Racket', 'Running Shoes',
            'Novel', 'Textbook', 'Magazine', 'Journal', 'Notebook',
            'Action Figure', 'Board Game', 'Puzzle', 'Doll', 'Toy Car',
            'Car Battery', 'Tires', 'Oil Filter', 'Brake Pads', 'Windshield Wiper',
            'Vitamins', 'Supplements', 'Thermometer', 'Blood Pressure Monitor', 'First Aid Kit',
            'Organic Food', 'Snacks', 'Beverages', 'Frozen Meals', 'Fresh Produce'
        ]
    
    def generate_products(self, num_products: int = 50) -> List[dict]:
        """Generate synthetic product data"""
        products = []
        for i in range(num_products):
            product = {
                'product_id': f'P{i+1:03d}',
                'product_name': random.choice(self.product_names) + f'_{random.randint(1, 100)}',
                'category': random.choice(self.categories),
                'price': round(random.uniform(10, 500), 2),
                'cost': round(random.uniform(5, 0.8 * round(random.uniform(10, 500), 2)), 2),  # Cost is typically lower than price
                'weight_kg': round(random.uniform(0.1, 10), 2),
                'seasonal_factor': random.uniform(0.5, 1.5),  # Seasonal demand factor
                'trend_factor': random.uniform(0.8, 1.2)      # Trend factor
            }
            products.append(product)
        
        return products
    
    def generate_sales_data(self, products: List[dict]) -> List[dict]:
        """Generate synthetic sales data with temporal patterns"""
        sales_data = []
        
        for product in products:
            for date in self.dates:
                # Base demand calculation with seasonal and trend factors
                base_demand = max(0, random.gauss(50, 20))  # Base demand with some randomness
                
                # Apply seasonal factor (higher during certain months)
                month_factor = 1.0
                if date.month in [11, 12]:  # Higher demand during holiday season
                    month_factor = 1.5
                elif date.month in [6, 7, 8]:  # Summer vacation period
                    month_factor = 0.8
                
                # Apply weekend effect
                day_factor = 1.0
                if date.weekday() >= 5:  # Weekend
                    day_factor = 1.2
                
                # Apply product-specific factors
                seasonal_effect = product['seasonal_factor']
                trend_effect = product['trend_factor']
                
                # Calculate final demand with noise
                demand = base_demand * month_factor * day_factor * seasonal_effect * trend_effect
                demand_with_noise = max(0, demand + random.gauss(0, demand * 0.1))
                
                # Convert demand to actual sales (with some probability of not selling)
                actual_sales = max(0, int(demand_with_noise * random.uniform(0.7, 1.3)))
                
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
                    'is_holiday_season': 1 if date.month in [11, 12] else 0
                }
                
                sales_data.append(sale_record)
        
        return sales_data
    
    def generate_inventory_data(self, products: List[dict]) -> List[dict]:
        """Generate synthetic inventory data"""
        inventory_data = []
        
        for product in products:
            # Random initial stock level
            initial_stock = random.randint(50, 500)
            
            for date in self.dates:
                # Simulate inventory changes
                stock_change = random.randint(-20, 20)  # Restocking or usage
                current_stock = max(0, initial_stock + stock_change)
                
                inventory_record = {
                    'date': date.strftime('%Y-%m-%d'),
                    'product_id': product['product_id'],
                    'current_stock': current_stock,
                    'reorder_level': random.randint(10, 30),
                    'reorder_quantity': random.randint(50, 100),
                    'lead_time_days': random.randint(2, 7)
                }
                
                inventory_data.append(inventory_record)
        
        return inventory_data
    
    def generate_customer_behavior(self, products: List[dict]) -> List[dict]:
        """Generate synthetic customer behavior data"""
        customer_data = []
        
        for product in products:
            for date in self.dates:
                # Customer engagement metrics
                customer_record = {
                    'date': date.strftime('%Y-%m-%d'),
                    'product_id': product['product_id'],
                    'page_views': max(0, int(random.gauss(100, 30))),
                    'add_to_cart': max(0, int(random.gauss(20, 10))),
                    'purchases': max(0, int(random.gauss(10, 5))),
                    'conversion_rate': random.uniform(0.05, 0.3),
                    'avg_rating': round(random.uniform(3.0, 5.0), 1),
                    'reviews_count': max(0, int(random.gauss(5, 3)))
                }
                
                customer_data.append(customer_record)
        
        return customer_data
    
    def generate_all_data(self) -> Tuple[List[dict], List[dict], List[dict]]:
        """Generate all synthetic datasets"""
        print("Generating synthetic products...")
        products = self.generate_products(num_products=50)
        
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
    sales_file_path = os.path.join(output_dir, "synthetic_sales_data.csv")
    with open(sales_file_path, 'w', newline='') as csvfile:
        if sales_data:
            fieldnames = sales_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sales_data)
    
    # Save inventory data
    inventory_file_path = os.path.join(output_dir, "synthetic_inventory_data.csv")
    with open(inventory_file_path, 'w', newline='') as csvfile:
        if inventory_data:
            fieldnames = inventory_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(inventory_data)
    
    # Save customer data
    customer_file_path = os.path.join(output_dir, "synthetic_customer_data.csv")
    with open(customer_file_path, 'w', newline='') as csvfile:
        if customer_data:
            fieldnames = customer_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(customer_data)
    
    print(f"Datasets saved to {output_dir}/ directory")


if __name__ == "__main__":
    # Generate synthetic data
    generator = SimpleSyntheticDataGenerator(start_date="2023-01-01", end_date="2024-12-31")
    sales_data, inventory_data, customer_data = generator.generate_all_data()
    
    # Save datasets
    save_datasets_as_csv(sales_data, inventory_data, customer_data)
    
    print("Synthetic data generation completed!")
    print(f"Sales data records: {len(sales_data)}")
    print(f"Inventory data records: {len(inventory_data)}")
    print(f"Customer data records: {len(customer_data)}")
    
    # Display sample of sales data
    print("\nSample of sales data:")
    if sales_data:
        print(sales_data[0])