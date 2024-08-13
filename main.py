from flask import Flask, request, jsonify
import os
import google.generativeai as genai

app = Flask(__name__)

# Configure Google Generative AI
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Initialize the model with the desired configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# In-memory storage for global conversation history
global_historique = []

def upload_to_gemini(file_path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(file_path, mime_type=mime_type)
    return file

@app.route('/api/', methods=['POST'])
def process_request():
    global global_historique

    # Check if 'file' is provided
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if not file.content_type.startswith('image/'):
            return jsonify({"error": "Invalid file type. Please upload an image."}), 400

        file_path = os.path.join('/tmp', file.filename)
        file.save(file_path)
        uploaded_file = upload_to_gemini(file_path, mime_type=file.content_type)
        os.remove(file_path)
        
        # Add the uploaded file to the conversation history
        global_historique.append({
            "role": "user",
            "parts": [uploaded_file, "Ce Quoi cette photo ?"]
        })
    
    question = request.form.get('question')
    if question:
        global_historique.append({
            "role": "user",
            "parts": [question]
        })
    
    # Start chat session with the historical context
    chat_session = model.start_chat(
        history=global_historique
    )
    
    response = chat_session.send_message(question)
    
    # Add the model's response to the conversation history
    global_historique.append({
        "role": "model",
        "parts": [response.text]
    })
    
    return jsonify({
        "response": response.text
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
