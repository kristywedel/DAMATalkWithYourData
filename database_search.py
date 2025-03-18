import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

# Initialize the SentenceTransformer model (use "all-MiniLM-L6-v2" for FAISS embedding)
encoder_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# FAISS Index initialization
embedding_size = 384  # Size of the output vector for the "all-MiniLM-L6-v2" model
index = faiss.IndexFlatL2(embedding_size)  # FAISS index for semantic search
product_ids = []  # To store product names (or IDs) for FAISS indexing

class DatabaseSearch:
    def __init__(self, db_path: str = "products.db"):
        self.db_path = db_path

    def search_products(self, query: str) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Search in products table using LIKE for basic text matching
        search_term = f"%{query}%"
        cursor.execute("""
            SELECT name, description, category 
            FROM products 
            WHERE name LIKE ? OR description LIKE ? OR category LIKE ?
        """, (search_term, search_term, search_term))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "name": row[0],
                "description": row[1],
                "category": row[2]
            })
        
        conn.close()
        return results

    def get_relationships(self, product_name: str, depth: int = 2) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("DROP TABLE IF EXISTS temp_graph")
        cursor.execute("""
            CREATE TEMPORARY TABLE temp_graph (
                entity TEXT,
                related_entity TEXT,
                relationship TEXT,
                depth INTEGER
            )
        """)
        
        # Initial relationships
        cursor.execute("""
            INSERT INTO temp_graph
            SELECT entity1, entity2, relationship, 1
            FROM relationships
            WHERE entity1 = ?
            UNION
            SELECT entity2, entity1, relationship, 1
            FROM relationships
            WHERE entity2 = ?
        """, (product_name, product_name))
        
        # Recursive relationships up to specified depth
        for d in range(2, depth + 1):
            cursor.execute("""
                INSERT INTO temp_graph
                SELECT g.entity, r.entity2, r.relationship, ?
                FROM temp_graph g
                JOIN relationships r ON g.related_entity = r.entity1
                WHERE g.depth = ? - 1
                UNION
                SELECT g.entity, r.entity1, r.relationship, ?
                FROM temp_graph g
                JOIN relationships r ON g.related_entity = r.entity2
                WHERE g.depth = ? - 1
            """, (d, d, d, d))
        
        # Get results with product information
        cursor.execute("""
            SELECT DISTINCT 
                g.entity, 
                g.related_entity, 
                g.relationship, 
                g.depth,
                p.name, 
                p.description, 
                p.category
            FROM temp_graph g
            LEFT JOIN products p ON g.related_entity = p.name
            ORDER BY g.depth
        """)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "entity": row["entity"],
                "related_entity": row["related_entity"],
                "relationship": row["relationship"],
                "depth": row["depth"],
                "product_name": row["name"],
                "product_description": row["description"],
                "product_category": row["category"]
            })
        
        conn.close()
        return results

# Function to load data from SQLite and initialize FAISS index
def load_data_from_db():
    global product_ids, index
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()
    
    # Load all product descriptions into FAISS
    cursor.execute("SELECT name, description FROM products")
    products = cursor.fetchall()
    
    product_ids = [product[0] for product in products]  # Store product names
    descriptions = [product[1] for product in products]  # Store descriptions

    # Encode descriptions for FAISS semantic search
    embeddings = encoder_model.encode(descriptions)
    embeddings = np.array(embeddings).astype("float32")

    # Add product embeddings to the FAISS index
    index.add(embeddings)
    conn.close()

# Function to search relevant products in SQLite based on a query
def search_in_sqlite(query, top_k=5):
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()
    
    # Perform an exact match or partial match in SQLite
    cursor.execute("SELECT name, description, category FROM products WHERE description LIKE ?", ('%' + query + '%',))
    results = cursor.fetchall()
    conn.close()
    
    return [{"name": result[0], "description": result[1], "category": result[2]} for result in results[:top_k]]

# Function to search for similar products in FAISS (semantic search)
def search_in_faiss(query, top_k=5):
    # Encode the query for semantic similarity
    query_embedding = encoder_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    
    # Search FAISS for the most similar products
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(product_ids):  # Ensure valid index
            results.append({"name": product_ids[idx], "similarity": float(1.0 / (1.0 + distances[0][i]))})  # Similarity score
    
    return results

# Function to search for related entities using recursive SQL queries
def search_relationships(entity, depth=2):
    conn = sqlite3.connect("products.db")
    conn.row_factory = sqlite3.Row  # To get dict-like rows
    cursor = conn.cursor()
    
    # Create a temporary table for recursive query
    cursor.execute("DROP TABLE IF EXISTS temp_graph")
    cursor.execute("""
        CREATE TEMPORARY TABLE temp_graph (
            entity TEXT,
            related_entity TEXT,
            relationship TEXT,
            depth INTEGER
        )
    """)
    
    # Seed with the initial entity
    cursor.execute("""
        INSERT INTO temp_graph
        SELECT entity1, entity2, relationship, 1
        FROM relationships
        WHERE entity1 = ? 
        UNION
        SELECT entity2, entity1, relationship, 1
        FROM relationships
        WHERE entity2 = ?
    """, (entity, entity))
    
    # Recursively find related entities up to specified depth
    for d in range(2, depth + 1):
        cursor.execute("""
            INSERT INTO temp_graph
            SELECT g.entity, r.entity2, r.relationship, ?
            FROM temp_graph g
            JOIN relationships r ON g.related_entity = r.entity1
            WHERE g.depth = ? - 1
            UNION
            SELECT g.entity, r.entity1, r.relationship, ?
            FROM temp_graph g
            JOIN relationships r ON g.related_entity = r.entity2
            WHERE g.depth = ? - 1
        """, (d, d, d, d))
    
    # Get all related entities with their products
    cursor.execute("""
        SELECT DISTINCT g.entity, g.related_entity, g.relationship, g.depth,
                        p.name, p.description, p.category
        FROM temp_graph g
        LEFT JOIN products p ON g.related_entity = p.name
        ORDER BY g.depth
    """)
    
    results = []
    for row in cursor.fetchall():
        results.append({
            "entity": row["entity"],
            "related_entity": row["related_entity"],
            "relationship": row["relationship"],
            "depth": row["depth"],
            "product_name": row["name"],
            "product_description": row["description"],
            "product_category": row["category"]
        })
    
    conn.close()
    return results

# Function to search the database for relevant results
def search(query, top_k=5, depth=2):
    # First, search in SQLite (structured product descriptions)
    sqlite_results = search_in_sqlite(query, top_k)
    
    # Then, search using FAISS (semantic similarity search)
    faiss_results = search_in_faiss(query, top_k)
    
    # Lastly, search for related entities using the recursive SQL relationships query
    relationships_results = search_relationships(query, depth)
    
    # Combine all the results (SQLite, FAISS, and relationships)
    return {
        "sqlite_results": sqlite_results,
        "faiss_results": faiss_results,
        "relationships_results": relationships_results
    }

# Call the load function to initialize the FAISS index when the script starts
load_data_from_db()

