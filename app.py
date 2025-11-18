import wfdb
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.signal import butter, filtfilt, find_peaks
from scipy import stats

# ---------------------------------------------------------
# 1. CREATE FLASK APP
# ---------------------------------------------------------
app = Flask(__name__)
CORS(app)  # This allows your HTML to send requests

# ---------------------------------------------------------
# 2. LOAD YOUR MODEL
# ---------------------------------------------------------
# Load the model you trained (make sure the .joblib file is in the same folder)
print("Loading global ML model...")
try:
    # Use the model you found was best (e.g., Random Forest)
    ml_model = joblib.load('rf_global_model.joblib')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("ERROR: Model file not found. Make sure 'rf_global_model.joblib' is in the folder.")
    ml_model = None

# ---------------------------------------------------------
# 3. COPY ALL YOUR PREPROCESSING FUNCTIONS
# (The backend MUST use the *exact* same functions as your training script)
# ---------------------------------------------------------
def bandpass_filter(sig, fs, lowcut=0.5, highcut=40, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, sig)

def remove_baseline(sig, fs, cutoff=0.5, order=3):
    nyq = fs / 2
    b, a = butter(order, cutoff/nyq, btype='high')
    return filtfilt(b, a, sig)

def lms_filter(signal, mu, n_taps, reference_freq, fs):
    t = np.arange(len(signal)) / fs
    reference_noise = np.sin(2 * np.pi * reference_freq * t)
    w = np.zeros(n_taps)
    denoised_signal = np.zeros_like(signal)
    padded_ref = np.pad(reference_noise, (n_taps - 1, 0), 'constant')

    for i in range(len(signal)):
        x = padded_ref[i:i+n_taps][::-1]
        y = np.dot(w, x)
        e = signal[i] - y
        w = w + 2 * mu * e * x
        denoised_signal[i] = e
    return denoised_signal

# ---------------------------------------------------------
# 4. DEFINE THE "BEATSENSE" ANALYSIS FUNCTION
# (This function will run your *entire* pipeline on a new record)
# ---------------------------------------------------------
def analyze_record(record_name):
    # --- 1. Load Data ---
    record = wfdb.rdrecord(record_name, pn_dir='mitdb/')
    ecg_signal = record.p_signal[:, 0]
    fs = record.fs
    time = np.arange(len(ecg_signal)) / fs
    
    # --- 2. Preprocessing ---
    filtered = bandpass_filter(ecg_signal, fs)
    baseline_removed = remove_baseline(filtered, fs)
    denoised = lms_filter(baseline_removed, mu=0.001, n_taps=30, reference_freq=60, fs=fs)
    
    # --- 3. Rhythm Analysis (Phase 5) ---
    r_peaks, _ = find_peaks(denoised, distance=int(0.2 * fs), height=0.5)
    r_r_intervals_sec = np.diff(r_peaks) / fs
    mean_hr = 60 / np.mean(r_r_intervals_sec)
    std_rr = np.std(r_r_intervals_sec)
    
    rhythm = "Normal Sinus Rhythm"
    rhythm_type = "normal"
    if mean_hr < 60:
        rhythm = "Bradycardia"
        rhythm_type = "brady"
    elif mean_hr > 100:
        if std_rr > 0.15:
            rhythm = "Tachycardia - Atrial Fibrillation"
        else:
            rhythm = "Tachycardia - Regular"
        rhythm_type = "tachy"

    # --- 4. Beat Analysis (Phase 4) ---
    if ml_model is None:
        return {"bpm": mean_hr, "rhythm": rhythm, "classification": "ML Model Not Loaded", "type": rhythm_type}
        
    pre = int(0.2 * fs)
    post = int(0.4 * fs)
    segment_len = pre + post
    beat_segments = []
    
    # Segment beats based on *our detected* R-peaks
    for idx in r_peaks:
        if idx - pre >= 0 and idx + post < len(denoised):
            beat = denoised[idx - pre : idx + post]
            if len(beat) == segment_len:
                beat_segments.append(beat)
    
    # Classify beats with our loaded model
    if len(beat_segments) > 0:
        predictions = ml_model.predict(beat_segments)
        # Find the most common beat type (the "mode")
        beat_classification = stats.mode(predictions)[0]
    else:
        beat_classification = "N/A"

    # --- 5. Combine Results ---
    final_classification = f"Rhythm: {rhythm} | Predominant Beat: {beat_classification}"
    if rhythm == "Tachycardia - Regular" and beat_classification == 'V':
        final_classification = "Tachycardia - Likely Ventricular Tachycardia"
    elif rhythm == "Tachycardia - Regular" and beat_classification == 'N':
        final_classification = "Tachycardia - Likely Supraventricular Tachycardia"

    return {
        "bpm": mean_hr,
        "rhythm": rhythm,
        "classification": final_classification,
        "type": rhythm_type
    }

# ---------------------------------------------------------
# 5. CREATE THE API ENDPOINT
# (This is the URL your HTML will call)
# ---------------------------------------------------------
@app.route('/analyze-mit', methods=['POST'])
def analyze_mit_endpoint():
    # Get the record name sent from the HTML
    data = request.get_json()
    record_name = data.get('record_name')
    
    if not record_name:
        return jsonify({"error": "No record name provided"}), 400
    
    print(f"Received request to analyze record: {record_name}")
    
    try:
        # Run our full analysis
        results = analyze_record(record_name)
        print(f"Analysis complete. Results: {results}")
        # Send the results back to the HTML
        return jsonify(results)
    except Exception as e:
        print(f"Error analyzing record {record_name}: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------
# 6. RUN THE SERVER
# ---------------------------------------------------------
if __name__ == '__main__':
    # This will run the server on http://127.0.0.1:5000
    app.run(debug=True, port=5000)
