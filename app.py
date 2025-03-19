from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from database_search import DatabaseSearch, search
from llm_model import LLMModel

# Initialize FastAPI app
app = FastAPI(title="Product Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db_search = DatabaseSearch()
llm = LLMModel()
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Pydantic models
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    redacted_query: str

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Product Chatbot API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                p { color: #666; }
            </style>
        </head>
        <body>
            <h1>Welcome to the Product Chatbot API</h1>
            <p>This API provides product information and recommendations using natural language queries.</p>
            <p>Use POST /chat endpoint to interact with the chatbot.</p>
        </body>
    </html>
    """

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Analyze and anonymize PII in the query
        analysis_results = analyzer.analyze(text=request.query, language="en")
        redacted_query = anonymizer.anonymize(text=request.query, analyzer_results=analysis_results).text

        # Search for relevant products and relationships
        search_results = search(redacted_query)
        
        # Extract products from search results
        products = search_results["sqlite_results"]
        relationships = search_results["relationships_results"]
        
        # Generate response using LLM
        response = llm.generate_response(
            query=redacted_query,
            products=products,
            relationships=relationships
        )

        # If no products found, provide a more informative response
        if not products:
            response = f"I apologize, but I couldn't find any specific information about {redacted_query} in our database. Would you like to try a different query or ask about our available products?"

        return ChatResponse(
            response=response,
            redacted_query=redacted_query
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True) 