import os
import tempfile
from flask import Flask, request, jsonify
import google.generativeai as genai
import uuid

# Configure the Google Generative AI SDK
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = Flask(__name__)

# Dictionary to store chat sessions
sessions = {}

def upload_to_gemini(file_path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(file_path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

# Model configuration
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

@app.route('/api/process', methods=['POST'])
def process_image_and_prompt():
    if 'image' not in request.files or 'prompt' not in request.form:
        return jsonify({"error": "Image and prompt are required."}), 400

    image = request.files['image']
    prompt = request.form['prompt']

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        image_path = temp_file.name
        image.save(image_path)

        # Upload the image to Gemini
        file_uri = upload_to_gemini(image_path, mime_type=image.mimetype)

        # Create or retrieve chat session
        session_id = str(uuid.uuid4())
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        file_uri,
                        prompt,
                    ],
                },
            ]
        )
        sessions[session_id] = chat_session

        response = chat_session.send_message(prompt)

    # Clean up temporary file
    os.remove(image_path)

    return jsonify({"session_id": session_id, "response": response.text})

@app.route('/api/query', methods=['GET'])
def query_prompt():
    session_id = request.args.get('session_id')
    prompt = request.args.get('prompt')
    
    if not session_id or not prompt:
        return jsonify({"error": "Session ID and prompt are required."}), 400

    chat_session = sessions.get(session_id)
    if not chat_session:
        return jsonify({"error": "Session not found."}), 404

    response = chat_session.send_message(prompt)
    return jsonify({"response": response.text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
