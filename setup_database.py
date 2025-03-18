import json
import sqlite3
from pathlib import Path

def create_database():
    """Create SQLite database and required tables"""
    conn = sqlite3.connect('products.db')
    cursor = conn.cursor()

    # Create products table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            description TEXT,
            category TEXT
        )
    ''')

    # Create relationships table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity1 TEXT,
            relationship TEXT,
            entity2 TEXT,
            UNIQUE(entity1, relationship, entity2)
        )
    ''')

    conn.commit()
    return conn, cursor

def load_data(cursor, conn):
    """Load data from data.json into the database"""
    try:
        with open('data.json', 'r') as file:
            data = json.load(file)
            
        # Insert products
        for product in data.get('products', []):
            cursor.execute('''
                INSERT OR REPLACE INTO products (name, description, category)
                VALUES (?, ?, ?)
            ''', (
                product.get('name', ''),
                product.get('description', ''),
                product.get('category', '')
            ))

        # Insert relationships
        for relation in data.get('relationships', []):
            cursor.execute('''
                INSERT OR REPLACE INTO relationships (entity1, relationship, entity2)
                VALUES (?, ?, ?)
            ''', (
                relation.get('entity1', ''),
                relation.get('relationship', ''),
                relation.get('entity2', '')
            ))

        conn.commit()
        return True
    except FileNotFoundError:
        print("Error: data.json file not found")
        return False
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in data.json")
        return False
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False

def main():
    # Create database and tables
    conn, cursor = create_database()
    
    # Load data from JSON
    success = load_data(cursor, conn)
    
    # Close database connection
    cursor.close()
    conn.close()
    
    if success:
        print("Database setup completed successfully!")
        print(f"Database file created at: {Path('products.db').absolute()}")
    else:
        print("Database setup failed. Please check the errors above.")

if __name__ == "__main__":
    main() 