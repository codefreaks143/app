import numpy as np
import tensorflow as tf
import librosa
import os
import tempfile
import logging
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load TFLite model with optional GPU delegate
def load_tflite_model():
    model_path = "models/music1_classifier.tflite"
    try:
        # Try loading GPU delegate
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_delegates=[tf.lite.experimental.load_delegate('libedgetpu.so.1')] if os.name != 'nt' else []
        )
        logger.info("✅ Loaded TFLite model with GPU delegate.")
    except Exception as e:
        logger.warning(f"⚠️ GPU delegate not available or failed to load: {e}")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        logger.info("💻 Loaded TFLite model on CPU.")

    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Get input/output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]
input_channels = input_details[0]['shape'][3]

# Class labels
class_labels = ["Over-stimulating", "Stimulating", "Non-stimulating"]

def classify_segment(segment, sr):
    """Classify a single audio segment."""
    # Generate Mel-spectrogram
    spectrogram = librosa.feature.melspectrogram(
        y=segment, sr=sr, n_mels=input_height, fmax=8000
    )
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Resize and normalize
    resized = tf.image.resize(spectrogram[..., np.newaxis], [input_height, input_width])
    min_val = tf.reduce_min(resized)
    max_val = tf.reduce_max(resized)
    normalized = (resized - min_val) / (max_val - min_val + 1e-6)
    input_tensor = np.expand_dims(normalized, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    if output_details[0]['dtype'] in [np.float32, np.float16]:
        prob = tf.nn.softmax(output).numpy()
    else:
        prob = output  # For quantized models

    return prob

@app.route('/classify', methods=['POST'])
def classify_audio():
    """Endpoint to classify an uploaded audio file."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Validate file type
    if not file.filename.lower().endswith(('.wav', '.mp3', '.ogg')):
        return jsonify({"error": "Invalid file type. Only WAV, MP3, and OGG are supported."}), 400

    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file_path = temp_file.name
            file.save(temp_file_path)

        # Load and resample audio
        audio, sr = librosa.load(temp_file_path, sr=22050)
        segment_duration = 10  # seconds
        samples_per_segment = segment_duration * sr
        total_segments = int(len(audio) // samples_per_segment)

        predictions = []

        for i in range(total_segments):
            start = i * samples_per_segment
            end = start + samples_per_segment
            segment = audio[start:end]

            # Classify each segment
            prob = classify_segment(segment, sr)
            predictions.append(prob)

        # Aggregate results
        avg_probs = np.mean(predictions, axis=0)
        predicted_index = np.argmax(avg_probs)
        predicted_label = class_labels[predicted_index]

        # Clean up temporary file
        os.remove(temp_file_path)

        return jsonify({
            "prediction": predicted_label,
            "confidence": float(avg_probs[predicted_index]),
            "probabilities": {class_labels[i]: float(p) for i, p in enumerate(avg_probs)}
        })

    except Exception as e:
        logger.error(f"Error during classification: {e}")
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)