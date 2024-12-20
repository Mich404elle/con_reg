from flask import Flask, request, jsonify, render_template
import openai
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from markdown import markdown
from bs4 import BeautifulSoup
import glob
import logging
import gc #garbage collection

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.debug = True

# Store conversation history
conversation_history = {}

# Set OpenAI API key directly
openai.api_key = os.getenv('OPENAI_API_KEY')

# Placeholder for document embeddings
documents = []

def strip_html_tags(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()

def load_markdown_files(data_folder="data"):
    logger.info(f"Loading markdown files from {data_folder}")
    os.makedirs(data_folder, exist_ok=True)
    markdown_files = glob.glob(os.path.join(data_folder, "*.md"))
    
    if not markdown_files:
        logger.info(f"No markdown files found. Creating sample file...")
        sample_file = os.path.join(data_folder, "sample.md")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("# Sample Document\n\nThis is a sample document created automatically.")
        markdown_files = [sample_file]
    
    for file_path in markdown_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                plain_text = strip_html_tags(markdown(content, output_format="html"))
                documents.append({
                    "content": plain_text, 
                    "file_path": file_path, 
                    "embedding": None
                })
                logger.info(f"Loaded: {file_path}")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")

def generate_embeddings(text):
    try:
        if not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
            
        response = openai.Embedding.create(
            input=text[:8191],  # Silently truncates text without warning
            model="text-embedding-ada-002"
        )
        return np.array(response["data"][0]["embedding"])
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return None

def precompute_embeddings():
    logger.info("Precomputing embeddings...")
    for doc in documents:
        if doc["embedding"] is None:
            doc["embedding"] = generate_embeddings(doc["content"])
            if doc["embedding"] is None:
                logger.warning(f"Failed to generate embedding for {doc['file_path']}")
            else:
                logger.info(f"Generated embedding for {doc['file_path']}")
            gc.collect()  # Force garbage collection after each embedding

def truncate_content(content, max_chars=16000):
    """Truncate content while preserving complete sentences"""
    if len(content) <= max_chars:
        return content
        
    truncated = content[:max_chars]
    # Try to end at a sentence boundary
    last_period = truncated.rfind('.')
    if last_period > 0:
        truncated = truncated[:last_period + 1]
    return truncated

# Load documents at startup
load_markdown_files()
precompute_embeddings()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        logger.info("Received chat request")
        
        if not openai.api_key:
            logger.error("OpenAI API key is not set")
            return jsonify({"error": "OpenAI API key is not configured"}), 500

        if not documents:
            logger.error("No documents loaded")
            return jsonify({"error": "No documents loaded"}), 500

        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({"error": "No JSON data provided"}), 400
            
        user_query = data.get("query", "").strip()
        session_id = data.get("session_id", "default")
        
        if not user_query:
            logger.error("No query provided")
            return jsonify({"error": "No query provided"}), 400

        logger.info(f"Processing query: {user_query}")

        # Get relevant document content
        query_embedding = generate_embeddings(user_query)
        if query_embedding is None:
            logger.error("Failed to generate query embedding")
            return jsonify({"error": "Failed to generate query embedding"}), 500

        valid_documents = [doc for doc in documents if doc["embedding"] is not None]
        if not valid_documents:
            logger.error("No valid documents to search")
            return jsonify({"error": "No valid documents to search"}), 500

        similarities = [
            cosine_similarity([query_embedding], [doc["embedding"]])[0][0]
            for doc in valid_documents
        ]
        best_match_index = int(np.argmax(similarities))
        best_document = valid_documents[best_match_index]
        
        # Limit the content length of the best document (approximately 4000 tokens)
        max_chars = 16000  # Approximate character limit (roughly 4000 tokens)
        truncated_content = best_document["content"][:max_chars]
        
        # Initialize or get conversation history
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        # Construct messages with truncated content
        messages = [
            {
                "role": "system",
                "content": """You are a architecture assistant to answer questions about the HongKong construction or architectural regulations.
                """ + truncated_content
            }
        ]
        
        # Add limited conversation history (last 5 messages)
        messages.extend(conversation_history[session_id][-5:])
        
        # Add current query
        messages.append({"role": "user", "content": user_query})
        
        # Get response from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        
        # Store the conversation (limit to last 5 messages)
        conversation_history[session_id].append({"role": "user", "content": user_query})
        conversation_history[session_id].append({"role": "assistant", "content": response.choices[0].message.content})
        
        if len(conversation_history[session_id]) > 10:
            conversation_history[session_id] = conversation_history[session_id][-10:]

        return jsonify({
            "response": response.choices[0].message.content.strip(),
            "source": os.path.basename(best_document['file_path'])
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_markdown_files()
    precompute_embeddings()
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=8080)