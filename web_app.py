#!/usr/bin/env python3
"""
Web Chat Interface for Llama 2 7B Models
Real-time chat with model switching and conversation history
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import json
import asyncio
import logging
import torch
from typing import Dict, List, Optional
from datetime import datetime
import uuid
from pathlib import Path

# Import our existing components
from utils import ModelLoader, setup_logging
from auth import verify_token

# Setup logging
logger = logging.getLogger(__name__)
setup_logging()

app = FastAPI(
    title="Llama 2 7B Chat Interface",
    description="Interactive web chat with quantized Llama models",
    version="1.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global model loader and current state
model_loader = ModelLoader()
current_model = None
current_tokenizer = None
current_model_name = None

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting Llama 2 7B Chat Interface")
    logger.info(f"Available models: {model_loader.list_available_models()}")
    
@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down Llama 2 7B Chat Interface")
    global current_model, current_tokenizer, current_model_name
    
    # Clear global references
    current_model = None
    current_tokenizer = None
    current_model_name = None
    
    # Force multiple rounds of garbage collection
    import gc
    for _ in range(5):
        gc.collect()
    
    # Unload all models
    try:
        unloaded_count = model_loader.unload_all_models()
        logger.info(f"Unloaded {unloaded_count} models during shutdown")
    except Exception as e:
        logger.error(f"Error during model cleanup: {e}")
    
    # Final GPU memory cleanup
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()
            logger.info("GPU memory cache cleared during shutdown")
        except Exception as e:
            logger.error(f"Error clearing GPU cache during shutdown: {e}")

class ConnectionManager:
    """Manages WebSocket connections for real-time chat"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.conversations: Dict[str, List[Dict]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Initialize conversation history if new client
        if client_id not in self.conversations:
            self.conversations[client_id] = []
        
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket, client_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_message(self, message: Dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    def add_to_conversation(self, client_id: str, message: Dict):
        if client_id not in self.conversations:
            self.conversations[client_id] = []
        
        self.conversations[client_id].append({
            **message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 50 messages to prevent memory issues
        if len(self.conversations[client_id]) > 50:
            self.conversations[client_id] = self.conversations[client_id][-50:]
    
    def get_conversation_history(self, client_id: str) -> List[Dict]:
        return self.conversations.get(client_id, [])

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """Main chat interface"""
    available_models = model_loader.list_available_models()
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "available_models": available_models,
        "title": "Llama 2 7B Chat"
    })

@app.get("/api/models")
async def get_available_models():
    """Get list of available models"""
    return {
        "models": model_loader.list_available_models(),
        "default": "llama2-7b-8bit"
    }

@app.get("/api/memory")
async def get_memory_info():
    """Get current memory usage information"""
    global current_model, current_tokenizer, current_model_name
    
    memory_usage = model_loader.get_current_memory_usage()
    
    return {
        "memory_usage": memory_usage,
        "loaded_models": list(model_loader.loaded_models.keys()),
        "current_global_model": current_model_name,
        "global_model_loaded": current_model is not None,
        "global_tokenizer_loaded": current_tokenizer is not None
    }

@app.post("/clear-memory")
async def clear_memory():
    """Manually clear all models from memory"""
    global current_model, current_tokenizer, current_model_name
    try:
        logger.info("Manual memory clear requested")
        
        # Clear global references first
        current_model = None
        current_tokenizer = None
        current_model_name = None
        
        # Force garbage collection
        import gc
        for _ in range(3):
            gc.collect()
        
        # Unload all models from the model loader
        unloaded_count = model_loader.unload_all_models()
        
        # Additional GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        memory_info = model_loader.get_current_memory_usage()
        
        return {
            "success": True,
            "message": f"Successfully cleared {unloaded_count} models from memory",
            "memory_info": memory_info
        }
    except Exception as e:
        logger.error(f"Failed to clear memory: {e}")
        return {"success": False, "error": str(e)}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket, client_id)
    
    try:
        # Send conversation history to reconnecting clients
        history = manager.get_conversation_history(client_id)
        if history:
            await manager.send_message({
                "type": "history",
                "messages": history
            }, websocket)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "chat":
                await handle_chat_message(websocket, client_id, message_data)
            elif message_data["type"] == "model_switch":
                await handle_model_switch(websocket, client_id, message_data)
            elif message_data["type"] == "clear_history":
                await handle_clear_history(websocket, client_id)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)

async def handle_chat_message(websocket: WebSocket, client_id: str, message_data: Dict):
    """Handle incoming chat messages and generate responses"""
    try:
        user_message = message_data.get("message", "").strip()
        model_type = message_data.get("model", "llama2-7b-8bit")
        
        if not user_message:
            return
        
        # Add user message to conversation
        user_msg = {
            "role": "user",
            "content": user_message,
            "model": model_type
        }
        manager.add_to_conversation(client_id, user_msg)
        
        # Send typing indicator
        await manager.send_message({
            "type": "typing",
            "message": "AI is thinking..."
        }, websocket)
        
        # Generate AI response
        try:
            # Load the specified model if not already loaded
            global current_model, current_tokenizer, current_model_name
            
            if not current_model or current_model_name != model_type:
                await manager.send_message({
                    "type": "status",
                    "message": f"Loading {model_type} model..."
                }, websocket)
                
                # Clear global references first if switching models
                if current_model is not None and current_model_name != model_type:
                    logger.info(f"Switching from {current_model_name} to {model_type}, clearing global references")
                    current_model = None
                    current_tokenizer = None
                    current_model_name = None
                    
                    # Force garbage collection before loading new model
                    import gc
                    for _ in range(3):
                        gc.collect()
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                
                # Load with automatic unloading of previous models
                current_model, current_tokenizer = model_loader.load_model(model_type, unload_previous=True)
                current_model_name = model_type
            
            # Generate response with conversation context
            conversation_history = manager.get_conversation_history(client_id)
            context = build_conversation_context(conversation_history, max_context=3)  # Reduced context
            
            # Create a more structured prompt
            if context:
                prompt = f"<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant. Provide concise, informative responses. Avoid repetition.<|eot_id|>\n\n{context}\n<|start_header_id|>user<|end_header_id|>\n{user_message}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
            else:
                prompt = f"<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant. Provide concise, informative responses. Avoid repetition.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n{user_message}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
            
            # Generate response using the model directly
            if not current_model or not current_tokenizer:
                raise Exception("Model not loaded properly")
            
            # Determine the device where the model is located
            model_device = next(current_model.parameters()).device
            
            inputs = current_tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=2048
            )
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            start_time = datetime.now()
            
            with torch.no_grad():
                outputs = current_model.generate(
                    **inputs,
                    max_new_tokens=512,  # Increased for complete responses
                    temperature=0.7,     # Lower temperature for more focused responses
                    top_p=0.9,           # Slightly lower for more focused responses
                    top_k=40,            # Reduced top-k
                    do_sample=True,
                    repetition_penalty=1.2,  # Higher penalty for repetition
                    pad_token_id=current_tokenizer.eos_token_id,
                    eos_token_id=current_tokenizer.eos_token_id,
                    early_stopping=True,
                    num_beams=1,
                    # Add custom stopping strings
                    stopping_criteria=None,  # We'll handle stopping in post-processing
                )
            
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            # Decode response
            generated_text = current_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the response more aggressively
            ai_response = generated_text.replace(prompt, "").strip()
            
            # Stop at common stopping patterns first
            stop_patterns = [
                "Would you like to know more",
                "If you have any other questions", 
                "What's on your mind",
                "Would you like to ask",
                "Let me know if you need",
                "Feel free to ask if",
                "What's next",
                "What can I help you with",
                "How can I assist you"
            ]
            
            # Find the first occurrence of any stop pattern and truncate there
            min_stop_pos = len(ai_response)
            for pattern in stop_patterns:
                pos = ai_response.find(pattern)
                if pos != -1 and pos < min_stop_pos:
                    min_stop_pos = pos
            
            if min_stop_pos < len(ai_response):
                ai_response = ai_response[:min_stop_pos].strip()
            
            # Remove repetitive sentences
            sentences = ai_response.split('. ')
            clean_sentences = []
            seen_sentences = set()
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Normalize sentence for comparison
                sentence_normalized = sentence.lower().replace('!', '').replace('?', '').strip()
                
                # Skip if we've seen this sentence or very similar
                if sentence_normalized in seen_sentences:
                    continue
                    
                # Skip if sentence is too generic or promotional
                if len(sentence) < 10 or any(phrase in sentence.lower() for phrase in [  # Reduced from 15 to 10
                    "i'm here to help", "feel free to ask", "let me know if you", "would you like to know more"  # More specific phrases
                ]):
                    continue
                
                clean_sentences.append(sentence)
                seen_sentences.add(sentence_normalized)
                
                # Allow more sentences for complete responses
                if len(clean_sentences) >= 5:  # Increased from 2 to 5
                    break
            
            # Reconstruct response
            if clean_sentences:
                ai_response = '. '.join(clean_sentences)
                if not ai_response.endswith('.') and not ai_response.endswith('!') and not ai_response.endswith('?'):
                    ai_response += '.'
            else:
                # If we filtered everything out, provide a minimal response
                ai_response = "I understand your question. Let me provide a direct answer."
            
            # Fallback: if response is too long, truncate at a reasonable length
            if len(ai_response) > 2000:  # Increased from 500 to 2000
                ai_response = ai_response[:2000].rsplit('.', 1)[0] + '.'
                
            # Fallback: if response is empty or too short, provide a basic response
            if not ai_response or len(ai_response.strip()) < 10:
                ai_response = "I understand your question, but I'm having trouble generating a clear response right now."
            
            # Calculate tokens generated (use actual tokens from output)
            tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            # Clean up the response
            if ai_response.startswith("Assistant:"):
                ai_response = ai_response[10:].strip()
            
            # Add AI response to conversation
            ai_msg = {
                "role": "assistant",
                "content": ai_response,
                "model": model_type,
                "tokens_generated": tokens_generated,
                "generation_time": generation_time,
                "tokens_per_second": tokens_per_second
            }
            manager.add_to_conversation(client_id, ai_msg)
            
            # Send response to client
            await manager.send_message({
                "type": "response",
                "message": ai_response,
                "model": model_type,
                "stats": {
                    "tokens": tokens_generated,
                    "time": generation_time,
                    "speed": tokens_per_second
                }
            }, websocket)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            await manager.send_message({
                "type": "error",
                "message": f"Sorry, I encountered an error: {str(e)}"
            }, websocket)
    
    except Exception as e:
        logger.error(f"Error handling chat message: {e}")
        await manager.send_message({
            "type": "error",
            "message": "Sorry, something went wrong processing your message."
        }, websocket)

async def handle_model_switch(websocket: WebSocket, client_id: str, message_data: Dict):
    """Handle model switching requests"""
    new_model = message_data.get("model", "llama2-7b-8bit")
    global current_model, current_tokenizer, current_model_name
    
    try:
        await manager.send_message({
            "type": "status",
            "message": f"Switching to {new_model}..."
        }, websocket)
        
        # Log current memory usage
        memory_before = model_loader.get_current_memory_usage()
        logger.info(f"Memory before switching: {memory_before}")
        
        # Clear global references first
        if current_model is not None:
            current_model = None
        if current_tokenizer is not None:
            current_tokenizer = None
        current_model_name = None
        
        # Force garbage collection before loading new model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Load the new model (this will automatically unload previous models)
        current_model, current_tokenizer = model_loader.load_model(new_model, unload_previous=True)
        current_model_name = new_model
        
        # Additional memory cleanup after loading new model
        import gc
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Log memory usage after switching
        memory_after = model_loader.get_current_memory_usage()
        logger.info(f"Memory after switching: {memory_after}")
        
        await manager.send_message({
            "type": "model_switched",
            "model": new_model,
            "message": f"Now using {new_model}",
            "memory_info": memory_after
        }, websocket)
        
        logger.info(f"Client {client_id} switched to model {new_model}")
        logger.info(f"Memory change: {memory_before} -> {memory_after}")
    
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        await manager.send_message({
            "type": "error",
            "message": f"Error switching to {new_model}: {str(e)}"
        }, websocket)

async def handle_clear_history(websocket: WebSocket, client_id: str):
    """Clear conversation history for a client"""
    try:
        manager.conversations[client_id] = []
        await manager.send_message({
            "type": "history_cleared",
            "message": "Conversation history cleared"
        }, websocket)
        
        logger.info(f"Cleared conversation history for client {client_id}")
    
    except Exception as e:
        logger.error(f"Error clearing history: {e}")

def build_conversation_context(conversation: List[Dict], max_context: int = 5) -> str:
    """Build conversation context from recent messages"""
    if not conversation:
        return ""
    
    # Get last max_context messages
    recent_messages = conversation[-max_context:]
    
    context_parts = []
    for msg in recent_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "user":
            context_parts.append(f"User: {content}")
        elif role == "assistant":
            context_parts.append(f"Assistant: {content}")
    
    return "\n".join(context_parts)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_connections": len(manager.active_connections),
        "available_models": model_loader.list_available_models(),
        "current_model": current_model_name
    }

if __name__ == "__main__":
    import uvicorn
    
    # Create required directories
    Path("templates").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    Path("static/css").mkdir(exist_ok=True)
    Path("static/js").mkdir(exist_ok=True)
    
    print("🚀 Starting Llama 2 7B Web Chat Interface...")
    print("📱 Open your browser to: http://localhost:8080")
    print("🤖 Available models: Original, 8-bit, 4-bit")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
