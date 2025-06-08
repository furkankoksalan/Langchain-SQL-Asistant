import pandas as pd
import sqlite3
import os
import glob
from pathlib import Path


def csv_to_sqlite(csv_files=None, output_db="my_custom_database.db"):
    """Convert CSV files to SQLite database"""

    print("CSV to SQLite Converter")
    print("=" * 30)

    if csv_files is None:
        csv_files = glob.glob("*.csv")

    if not csv_files:
        print("No CSV files found!")
        print("Creating sample CSV files for testing...")
        create_sample_csv_files()
        csv_files = ["employees.csv", "departments.csv", "sales.csv"]

    print(f"Found {len(csv_files)} CSV files: {csv_files}")

    # Remove existing database
    if os.path.exists(output_db):
        os.remove(output_db)
        print(f"Removed existing {output_db}")

    # Create connection
    conn = sqlite3.connect(output_db)
    print(f"Created database: {output_db}")

    total_rows = 0

    for csv_file in csv_files:
        try:
            # Check if file exists
            if not os.path.exists(csv_file):
                print(f"Warning: File not found: {csv_file}")
                continue

            print(f"\nProcessing: {csv_file}")

            # Read CSV
            df = pd.read_csv(csv_file)

            # Clean table name
            table_name = Path(csv_file).stem.lower()
            table_name = table_name.replace(' ', '_').replace('-', '_')

            # Clean column names
            original_columns = df.columns.tolist()
            df.columns = [col.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '') for col in
                          df.columns]

            # Write to SQLite
            df.to_sql(table_name, conn, index=False, if_exists='replace')

            total_rows += len(df)

            print(f"   Table: {table_name}")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {list(df.columns)}")

            # Show sample data
            if len(df) > 0:
                print(f"   Sample data:")
                for i, row in df.head(2).iterrows():
                    print(f"      Row {i + 1}: {dict(row)}")

        except Exception as e:
            print(f"   Error processing {csv_file}: {str(e)}")

    conn.close()

    print(f"\nDatabase created successfully!")
    print(f"Total rows processed: {total_rows}")
    print(f"Database file: {output_db}")
    print(f"Upload this file in Streamlit!")

    return output_db


def create_sample_csv_files():
    """Create sample CSV files for testing"""

    print("Creating sample CSV files...")

    # Sample employees data
    employees_data = {
        'Employee ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'First Name': ['John', 'Jane', 'Mike', 'Sarah', 'David', 'Lisa', 'Tom', 'Anna', 'Mark', 'Emily'],
        'Last Name': ['Smith', 'Johnson', 'Brown', 'Davis', 'Wilson', 'Garcia', 'Martinez', 'Anderson', 'Taylor',
                      'Thomas'],
        'Department': ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Engineering', 'Sales', 'Marketing', 'IT',
                       'Operations'],
        'Position': ['Developer', 'Manager', 'Representative', 'Specialist', 'Analyst', 'Senior Developer', 'Manager',
                     'Coordinator', 'Admin', 'Supervisor'],
        'Salary': [75000, 85000, 55000, 60000, 70000, 95000, 78000, 52000, 65000, 72000],
        'Hire Date': ['2020-01-15', '2019-03-20', '2021-06-10', '2020-09-05', '2018-11-12', '2017-05-08', '2021-02-14',
                      '2022-01-20', '2019-08-30', '2020-12-03'],
        'Performance Score': [4.2, 4.8, 3.9, 4.1, 4.5, 4.9, 4.0, 3.8, 4.3, 4.2]
    }

    # Sample departments data
    departments_data = {
        'Department ID': [1, 2, 3, 4, 5, 6, 7],
        'Department Name': ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'IT', 'Operations'],
        'Manager': ['Lisa Garcia', 'Jane Johnson', 'Tom Martinez', 'Sarah Davis', 'David Wilson', 'Mark Taylor',
                    'Emily Thomas'],
        'Budget': [500000, 200000, 300000, 150000, 250000, 180000, 220000],
        'Location': ['San Francisco', 'New York', 'Chicago', 'Austin', 'Boston', 'Seattle', 'Denver'],
        'Employee Count': [25, 12, 18, 8, 10, 15, 12]
    }

    # Sample sales data
    sales_data = {
        'Sale ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'Employee ID': [3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7],
        'Product': ['Software License', 'Consulting', 'Training', 'Support', 'Software License', 'Consulting',
                    'Training', 'Support', 'Software License', 'Consulting', 'Training', 'Support'],
        'Sale Date': ['2023-01-15', '2023-01-20', '2023-02-05', '2023-02-12', '2023-03-08', '2023-03-15', '2023-04-02',
                      '2023-04-18', '2023-05-05', '2023-05-22', '2023-06-10', '2023-06-25'],
        'Amount': [50000, 75000, 25000, 35000, 60000, 80000, 30000, 40000, 55000, 70000, 28000, 38000],
        'Customer': ['TechCorp', 'DataSoft', 'AI Solutions', 'CloudWorks', 'StartupX', 'DevTools Inc', 'Analytics Pro',
                     'Code Masters', 'Tech Innovators', 'Digital Solutions', 'Smart Systems', 'Future Tech'],
        'Region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West', 'North', 'South', 'East', 'West']
    }

    # Create CSV files
    pd.DataFrame(employees_data).to_csv('employees.csv', index=False)
    pd.DataFrame(departments_data).to_csv('departments.csv', index=False)
    pd.DataFrame(sales_data).to_csv('sales.csv', index=False)

    print("Created employees.csv (10 employees)")
    print("Created departments.csv (7 departments)")
    print("Created sales.csv (12 sales records)")


def analyze_csv_files():
    """Analyze CSV files before conversion"""

    csv_files = glob.glob("*.csv")

    if not csv_files:
        print("No CSV files found!")
        return

    print("CSV Files Analysis:")
    print("=" * 30)

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            print(f"\n{csv_file}:")
            print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Data types: {dict(df.dtypes)}")

            # Check for missing values
            missing = df.isnull().sum()
            if missing.any():
                print(f"   Missing values: {dict(missing[missing > 0])}")
            else:
                print("   No missing values")

        except Exception as e:
            print(f"   Error reading {csv_file}: {str(e)}")


def main():
    """Main function with user interaction"""

    print("CSV to SQLite Converter")
    print("=" * 40)

    # Check for existing CSV files
    csv_files = glob.glob("*.csv")

    if csv_files:
        print(f"Found {len(csv_files)} CSV files:")
        for i, file in enumerate(csv_files, 1):
            print(f"   {i}. {file}")

        print("\nOptions:")
        print("1. Convert all CSV files to SQLite")
        print("2. Analyze CSV files first")
        print("3. Create sample CSV files and convert")

        try:
            choice = input("\nEnter your choice (1-3): ").strip()

            if choice == "1":
                output_file = csv_to_sqlite(csv_files)
                print(f"\nNext step: Upload '{output_file}' in Streamlit!")

            elif choice == "2":
                analyze_csv_files()
                if input("\nConvert now? (y/n): ").lower().startswith('y'):
                    output_file = csv_to_sqlite(csv_files)
                    print(f"\nNext step: Upload '{output_file}' in Streamlit!")

            elif choice == "3":
                output_file = csv_to_sqlite()  # Will create samples
                print(f"\nNext step: Upload '{output_file}' in Streamlit!")

            else:
                print("Invalid choice!")

        except KeyboardInterrupt:
            print("\nCancelled by user")

    else:
        print("No CSV files found. Creating sample files...")
        output_file = csv_to_sqlite()
        print(f"\nNext step: Upload '{output_file}' in Streamlit!")


if __name__ == "__main__":
    main()