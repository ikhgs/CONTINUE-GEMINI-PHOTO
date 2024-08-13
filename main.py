import os
import tempfile
from flask import Flask, request, jsonify
import google.generativeai as genai

# Configure the Google Generative AI SDK
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = Flask(__name__)

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

# Global history to track the last conversation
global_chat_session = model.start_chat(history=[])

@app.route('/api/process', methods=['POST'])
def process_image_and_prompt():
    if 'image' not in request.files or 'prompt' not in request.form:
        return jsonify({"error": "Image and prompt are required."}), 400

    prompt = request.form['prompt']

    # Handle conversation reset
    if prompt.lower() == 'stop':
        global global_chat_session
        global_chat_session = model.start_chat(history=[])
        return jsonify({"response": "Conversation has been reset."})

    # Save the image temporarily
    image = request.files['image']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        image_path = temp_file.name
        image.save(image_path)

        # Upload the image to Gemini
        file_uri = genai.upload_file(image_path, mime_type=image.mimetype).uri

        # Update the global chat session with the image and prompt
        global_chat_session.history.append({
            "role": "user",
            "parts": [file_uri, prompt],
        })

        # Send the message and get the response
        response = global_chat_session.send_message(prompt)

    # Clean up temporary file
    os.remove(image_path)

    return jsonify({"response": response.text})

@app.route('/api/query', methods=['GET'])
def query_prompt():
    prompt = request.args.get('prompt')

    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    # Continue the conversation from where it was left off in the POST request
    global_chat_session.history.append({
        "role": "user",
        "parts": [prompt],
    })

    # Send the message and get the response
    response = global_chat_session.send_message(prompt)
    return jsonify({"response": response.text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
