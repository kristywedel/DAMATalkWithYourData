from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LLMModel:
    def __init__(self, model_name: str = "facebook/opt-125m"):
        # Using a smaller model for testing - replace with your preferred model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=200,
            temperature=0.7
        )

    def generate_response(self, 
                         query: str, 
                         products: List[Dict[str, Any]], 
                         relationships: List[Dict[str, Any]]) -> str:
        # Create context from products and relationships
        context = f"Query: {query}\n\nRelevant products:\n"
        
        for i, product in enumerate(products[:3], 1):
            context += f"{i}. {product['name']} ({product['category']}): {product['description']}\n"
        
        if relationships:
            context += "\nRelated products:\n"
            seen = set()
            for rel in relationships[:3]:
                if rel['related_entity'] not in seen:
                    context += f"- {rel['entity']} {rel['relationship']} {rel['related_entity']}\n"
                    seen.add(rel['related_entity'])
        
        prompt = f"""
        Based on the following information, provide a helpful response to the user's query:
        
        {context}
        
        Response:"""
        
        # Generate response
        response = self.generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
        
        # Extract the response part (after "Response:")
        response = response.split("Response:")[-1].strip()
        
        return response
