from flask import Flask, render_template, request, jsonify
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)

# --- 1. GLOBAL SETTINGS & DUMMY MODEL TRAINING ---
# Realistically, we train the model once and save it. 
# For this demo, we will train a small model on startup using one record.
print("--- System Startup: Initializing AI Model ---")

# Placeholder model
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)

# We need to "fake" train it so it doesn't crash if you haven't downloaded data yet.
# In a real app, you would load a saved .pkl file here.
X_dummy = np.random.rand(10, 216) # 10 beats, 216 samples long
y_dummy = ['N', 'N', 'A', 'N', 'V', 'N', 'R', 'N', 'N', 'L']
rf_model.fit(X_dummy, y_dummy)

print("--- AI Model Ready ---")

# --- 2. HELPER FUNCTIONS (Signal Processing) ---
def process_signal(sig, fs=360):
    # Bandpass Filter
    nyq = fs / 2
    b, a = butter(4, [0.5/nyq, 40/nyq], btype='band')
    filtered = filtfilt(b, a, sig)
    return filtered

# --- 3. WEB ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze_record', methods=['POST'])
def analyze_record():
    data = request.json
    record_name = data.get('record_id')
    
    print(f"Received request to analyze Record {record_name}")
    
    try:
        # Download/Read the record from PhysioNet
        # This saves it to a local folder so we don't download it twice
        record = wfdb.rdrecord(record_name, pn_dir='mitdb')
        signal = record.p_signal[:, 0] # Lead I
        
        # Process data
        clean_signal = process_signal(signal)
        
        # Calculate Heart Rate (BPM)
        peaks, _ = find_peaks(clean_signal, distance=150) # Distance for 360Hz
        bpm = len(peaks) / (len(clean_signal) / 360) * 60
        
        # Grab a 3-second snapshot for the graph (first 1000 samples)
        # We limit data size so the website doesn't lag
        display_data = clean_signal[0:1000].tolist() 
        
        # Mock Prediction (Since we are just demoing the connection)
        # In the real version, you'd segment 'clean_signal' and feed to 'rf_model'
        if int(record_name) > 200:
            diagnosis = "Abnormal Arrhythmia Detected"
            rhythm = "Premature Ventricular Contraction (PVC)"
            rec_title = "Consult Cardiologist"
            rec_text = "Irregular ventricular activity detected. Clinical correlation recommended."
        else:
            diagnosis = "Normal Sinus Rhythm"
            rhythm = "Normal (N)"
            rec_title = "Routine Checkup"
            rec_text = "Heart rhythm appears within normal limits. Maintain healthy lifestyle."

        return jsonify({
            "success": True,
            "bpm": int(bpm),
            "rhythm": rhythm,
            "diagnosis": diagnosis,
            "recommendation_title": rec_title,
            "recommendation_text": rec_text,
            "signal_data": display_data # Sending data to draw the graph
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
