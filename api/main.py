import os
import tempfile
import requests
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

# Global chat history to maintain conversation state
chat_history = []

@app.route('/api/process', methods=['POST'])
def process_image_and_prompt():
    image_url = request.form.get('image_url')
    image_file = request.files.get('image')
    prompt = request.form.get('prompt')

    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    if image_url:
        # Process image URL
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download image from URL."}), 400

        # Save the image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(response.content)
            image_path = temp_file.name

    elif image_file:
        # Process uploaded image file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            image_path = temp_file.name
            image_file.save(image_path)

    else:
        return jsonify({"error": "Either 'image_url' or 'image' is required."}), 400

    # Upload the image to Gemini
    file_uri = genai.upload_file(image_path, mime_type="image/jpeg").uri

    # Update the chat history with the image and prompt
    chat_history.append({
        "role": "user",
        "parts": [file_uri, prompt],
    })

    # Send the message and get the response
    response = model.send_message(prompt, history=chat_history)

    # Clean up temporary file
    os.remove(image_path)

    return jsonify({"response": response.text})

@app.route('/api/query', methods=['GET'])
def query_prompt():
    prompt = request.args.get('prompt')

    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    # Add the prompt to the chat history
    chat_history.append({
        "role": "user",
        "parts": [prompt],
    })

    # Send the message and get the response
    response = model.send_message(prompt, history=chat_history)
    
    return jsonify({"response": response.text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
