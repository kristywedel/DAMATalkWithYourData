import json
import sqlite3
import faiss
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Global variables for singleton pattern
_model = None
_tokenizer = None
_embedding_model = None
_faiss_index = None
_product_ids = None

def get_embedding_model():
    """Initialize and return the sentence transformer model"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _embedding_model

def get_llm():
    """Initialize and return the LLM model and tokenizer"""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        model_name = "openchat/openchat-3.5"
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    return _model, _tokenizer

def initialize_faiss():
    """Initialize FAISS index with product descriptions"""
    global _faiss_index, _product_ids
    
    if _faiss_index is not None:
        return _faiss_index, _product_ids

    # Connect to SQLite database
    conn = sqlite3.connect('products.db')
    cursor = conn.cursor()
    
    # Get all products
    cursor.execute('SELECT id, description FROM products')
    products = cursor.fetchall()
    
    # Get embeddings for descriptions
    embedding_model = get_embedding_model()
    descriptions = [prod[1] for prod in products]
    embeddings = embedding_model.encode(descriptions)
    
    # Initialize FAISS index
    dimension = embeddings.shape[1]
    _faiss_index = faiss.IndexFlatL2(dimension)
    _faiss_index.add(np.array(embeddings).astype('float32'))
    
    # Store product IDs for later reference
    _product_ids = [prod[0] for prod in products]
    
    conn.close()
    return _faiss_index, _product_ids

def get_related_entities(cursor, entity: str, depth: int = 2) -> List[Dict]:
    """Recursively query related entities from the database"""
    if depth <= 0:
        return []

    results = []
    
    # Query both directions in relationships table
    cursor.execute('''
        SELECT entity1, relationship, entity2 
        FROM relationships 
        WHERE entity1 = ? OR entity2 = ?
    ''', (entity, entity))
    
    relations = cursor.fetchall()
    
    for rel in relations:
        entity1, relationship, entity2 = rel
        related_entity = entity2 if entity1 == entity else entity1
        
        # Add current relationship
        results.append({
            'entity': related_entity,
            'relationship': relationship,
            'depth': depth
        })
        
        # Recursively get related entities
        nested_relations = get_related_entities(cursor, related_entity, depth - 1)
        results.extend(nested_relations)
    
    return results

def redact_pii(text: str) -> str:
    """Redact PII from text using Presidio"""
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    
    # Analyze text for PII
    results = analyzer.analyze(text=text, language='en')
    
    # Anonymize identified PII
    anonymized_text = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    ).text
    
    return anonymized_text

def query_faiss(query: str, top_k: int = 3) -> List[Dict]:
    """Query FAISS index for similar products"""
    # Initialize if not already done
    faiss_index, product_ids = initialize_faiss()
    
    # Get query embedding
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode([query])
    
    # Search FAISS index
    distances, indices = faiss_index.search(
        np.array(query_embedding).astype('float32'), 
        top_k
    )
    
    # Get product details from database
    conn = sqlite3.connect('products.db')
    cursor = conn.cursor()
    
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        product_id = product_ids[idx]
        
        # Get product details
        cursor.execute('SELECT name, description, category FROM products WHERE id = ?', 
                      (product_id,))
        product = cursor.fetchone()
        
        if product:
            results.append({
                'name': product[0],
                'description': product[1],
                'category': product[2],
                'similarity_score': float(1 / (1 + distance))
            })
    
    conn.close()
    return results

def generate_response(context: Dict, query: str) -> str:
    """Generate a response using the LLM based on context and query"""
    model, tokenizer = get_llm()
    
    # Redact any PII from the query
    safe_query = redact_pii(query)
    
    # Construct prompt with context
    prompt = f"""Based on the following context, please answer the user's question.
    
Context:
- Related Products: {context.get('products', [])}
- Related Entities: {context.get('related_entities', [])}

User Question: {safe_query}

Please provide a helpful and informative response based on the available product information and relationships.

Response:"""

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response.split("Response:")[-1].strip()

def get_context(query: str) -> Dict:
    """Aggregate context from FAISS and database for a query"""
    # Get similar products
    similar_products = query_faiss(query)
    
    # Get related entities for each product
    conn = sqlite3.connect('products.db')
    cursor = conn.cursor()
    
    related_entities = []
    for product in similar_products:
        entities = get_related_entities(cursor, product['name'])
        related_entities.extend(entities)
    
    conn.close()
    
    return {
        'products': similar_products,
        'related_entities': related_entities
    } 