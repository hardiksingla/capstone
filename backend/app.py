import sys
import os
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent dir so we can import the predictor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from predictor_final import LinearTypingDetector

app = Flask(__name__)
CORS(app)

# Load model once at startup
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
detector = LinearTypingDetector(model_dir=MODEL_DIR)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.endswith('.json'):
        return jsonify({'error': 'File must be a JSON file'}), 400

    # Save to temp file and analyze
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    try:
        file.save(tmp.name)
        result = detector.predict_session(tmp.name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp.name)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
