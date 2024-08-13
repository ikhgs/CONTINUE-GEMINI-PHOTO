import os
import tempfile
from flask import Flask, request, jsonify
import google.generativeai as genai
import uuid

# Configure the Google Generative AI SDK
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = Flask(__name__)

# Dictionary to store chat sessions and history
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

        # Create a unique session ID
        session_id = str(uuid.uuid4())

        # Start a new chat session with the image and prompt
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

        # Store the session in memory
        sessions[session_id] = {
            'chat_session': chat_session,
            'history': [
                {
                    "role": "user",
                    "parts": [
                        file_uri,
                        prompt,
                    ],
                },
            ]
        }

    # Clean up temporary file
    os.remove(image_path)

    return jsonify({"session_id": session_id, "response": "Image and prompt processed. You can continue the conversation with GET requests."})

@app.route('/api/query', methods=['GET'])
def query_prompt():
    session_id = request.args.get('session_id')
    prompt = request.args.get('prompt')

    if not session_id or not prompt:
        return jsonify({"error": "Session ID and prompt are required."}), 400

    session_data = sessions.get(session_id)
    if not session_data:
        return jsonify({"error": "Session not found."}), 404

    chat_session = session_data['chat_session']
    history = session_data['history']

    # Add the new prompt to the history
    history.append({
        "role": "user",
        "parts": [prompt],
    })

    # Send the message to the model
    response = chat_session.send_message(prompt)

    # Update the session history
    session_data['history'] = history

    return jsonify({"response": response.text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
