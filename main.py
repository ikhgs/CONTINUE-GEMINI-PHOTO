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

def upload_to_gemini(file_path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(file_path, mime_type=mime_type)
    return file

@app.route('/api/', methods=['POST'])
def process_request():
    # Check if an image file and a question are provided
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    if 'question' not in request.form:
        return jsonify({"error": "No question provided"}), 400

    file = request.files['file']
    question = request.form['question']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file temporarily
    file_path = os.path.join('/tmp', file.filename)
    file.save(file_path)

    # Upload the file to Gemini
    uploaded_file = upload_to_gemini(file_path, mime_type=file.content_type)

    # Start chat session
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    uploaded_file,
                    "Ce Quoi cette photo ?",
                ],
            },
            {
                "role": "model",
                "parts": [
                    "C'est une photo d'Andry Rajoelina, le président de Madagascar. Il est connu pour avoir dirigé le pays de 2018 à 2022.",
                ],
            },
        ]
    )

    # Send the user question and get a response
    response = chat_session.send_message(question)

    # Clean up the temporary file
    os.remove(file_path)

    return jsonify({"response": response.text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
