from flask import Flask, request, jsonify
import os
import requests
import google.generativeai as genai
import uuid

# Configure Flask app
app = Flask(__name__)

# Configure Google AI SDK
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# In-memory storage for conversations (for demonstration purposes)
conversations = {}

# Upload function to Gemini
def upload_to_gemini(path, mime_type=None):
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

# Create the model
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

# Define the API route
@app.route('/api', methods=['GET'])
def api():
    # Get the parameters from the request
    image_url = request.args.get('image')
    user_message = request.args.get('message')
    session_id = request.args.get('session_id')

    if image_url:
        # Handle starting a new conversation
        try:
            # Download the image
            response = requests.get(image_url, stream=True)
            response.raise_for_status()  # Check if the request was successful
            
            file_path = os.path.join("/tmp", "downloaded_image")
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Upload the image to Gemini
            uploaded_file = upload_to_gemini(file_path)

            # Create a new chat session
            session_id = str(uuid.uuid4())
            chat_session = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [
                            uploaded_file,
                            "Faire cette Exercice ",
                        ],
                    },
                    {
                        "role": "model",
                        "parts": [
                            "Je suis prêt à répondre à vos questions concernant l'image.",
                        ],
                    },
                ]
            )

            # Store the session
            conversations[session_id] = chat_session

            return jsonify({"session_id": session_id, "message": "Conversation started"}), 200

        except requests.RequestException as e:
            return jsonify({"error": f"Failed to download image: {str(e)}"}), 400

    elif user_message and session_id:
        # Handle continuing an existing conversation
        chat_session = conversations.get(session_id)
        if not chat_session:
            return jsonify({"error": "Invalid session ID"}), 400

        # Send the user message to the chat session
        response = chat_session.send_message(user_message)
        
        return jsonify({"response": response.text})

    return jsonify({"error": "Invalid parameters"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
