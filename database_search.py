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
    print("\nInitializing FAISS index...")
    
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()
    
    # Load all product descriptions into FAISS
    cursor.execute("SELECT name, description FROM products")
    products = cursor.fetchall()
    
    if not products:
        print("Warning: No products found in database")
        return
    
    print(f"Found {len(products)} products in database")
    
    product_ids = [product[0] for product in products]  # Store product names
    descriptions = [product[1] for product in products]  # Store descriptions

    # Encode descriptions for FAISS semantic search
    print("Encoding product descriptions...")
    embeddings = encoder_model.encode(descriptions)
    embeddings = np.array(embeddings).astype("float32")
    print(f"Encoded {embeddings.shape[0]} descriptions with {embeddings.shape[1]} dimensions")

    # Reset and reinitialize the FAISS index
    global index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    print(f"Initialized FAISS index with {len(product_ids)} products")
    print(f"Index size: {index.ntotal} vectors")
    conn.close()

# Function to search relevant products in SQLite based on a query
def search_in_sqlite(query, top_k=5):
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()
    
    # Search across name, description, and category fields
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
        print(f"SQLite found: {row[0]} ({row[2]})")
    
    conn.close()
    print(f"SQLite search found {len(results)} results")
    return results[:top_k]

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
            name = product_ids[idx]
            similarity = float(1.0 / (1.0 + distances[0][i]))
            results.append({"name": name, "similarity": similarity})
            print(f"FAISS found: {name} (similarity: {similarity:.3f})")
    
    print(f"FAISS search found {len(results)} results")
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
    print(f"\nSearching for: {query}")
    
    # Semantic search using FAISS
    query_embedding = encoder_model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k=top_k)
    
    semantic_results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(product_ids):
            name = product_ids[idx]
            similarity = float(1.0 / (1.0 + distances[0][i]))
            semantic_results.append({
                "name": name,
                "similarity": similarity
            })
    
    # SQL search for exact matches
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()
    
    search_term = f"%{query}%"
    cursor.execute("""
        SELECT name, description, category 
        FROM products 
        WHERE name LIKE ? OR description LIKE ?
    """, (search_term, search_term))
    
    sql_results = []
    for row in cursor.fetchall():
        sql_results.append({
            "name": row[0],
            "description": row[1],
            "category": row[2]
        })
    
    conn.close()
    
    # Get related entities for the most relevant product
    relationships = []
    if semantic_results:
        relationships = search_relationships(semantic_results[0]["name"], depth)
    
    # Combine and deduplicate results
    seen_names = set()
    combined_results = []
    
    # Add semantic results first (they're more relevant)
    for result in semantic_results:
        if result["name"] not in seen_names:
            seen_names.add(result["name"])
            # Get full product details
            conn = sqlite3.connect("products.db")
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name, description, category 
                FROM products 
                WHERE name = ?
            """, (result["name"],))
            row = cursor.fetchone()
            if row:
                combined_results.append({
                    "name": row[0],
                    "description": row[1],
                    "category": row[2],
                    "similarity": result["similarity"]
                })
            conn.close()
    
    # Add SQL results that weren't already included
    for result in sql_results:
        if result["name"] not in seen_names:
            seen_names.add(result["name"])
            combined_results.append(result)
    
    # Sort by similarity if available
    combined_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    
    return {
        "sqlite_results": combined_results[:top_k],
        "relationships_results": relationships
    }

# Call the load function to initialize the FAISS index when the script starts
print("Starting database initialization...")
load_data_from_db()
print("Database initialization complete")

