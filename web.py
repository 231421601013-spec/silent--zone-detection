from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image
import os
from werkzeug.utils import secure_filename
import pandas as pd
import io
 
app = Flask(__name__)
app.secret_key = 'silent-zone-secret-key-2026-change-for-production'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
 
# ============================================
# LOAD ALL 4 MODELS + ENCODERS
# ============================================
ais_model        = joblib.load("ais_model.pkl")
le_vessel        = joblib.load("le_vessel.pkl")
le_cargo         = joblib.load("le_cargo.pkl")
le_status        = joblib.load("le_status.pkl")
vessel_map       = joblib.load("vessel_map.pkl")
 
ocean_model      = joblib.load("ocean_model.pkl")
ocean_threshold  = joblib.load("ocean_threshold.pkl")   # 0.35
 
marine_model     = joblib.load("marine_model.pkl")
marine_le        = joblib.load("marine_label_encoder.pkl")
 
whale_model      = load_model("whale_model.keras", compile=False)
 
# ============================================
# LOGIN
# ============================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['user_id']   = request.form['email']
        session['logged_in'] = True
        return redirect(url_for('upload'))
    return render_template('login.html')
 
# ============================================
# UPLOAD + PREDICT
# ============================================
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
 
    if request.method == 'POST':
 
        # ---------- 1. FORM INPUTS ----------
        pH          = float(request.form['pH'])
        oxygen      = float(request.form['oxygen'])
        turbidity   = float(request.form['turbidity'])
        nitrate     = float(request.form.get('nitrate', 5.0))
        temperature = float(request.form.get('temperature', 28.0))
        vessel_type = request.form['vessel_type']   # e.g. "cargo", "fishing", "tanker"
 
        # ---------- 2. OCEAN MODEL ----------
        # Features used in training: temperature, pH, dissolved_oxygen, turbidity, nitrate
        ocean_input = np.array([[temperature, pH, oxygen, turbidity, nitrate]])
        ocean_prob  = ocean_model.predict_proba(ocean_input)[0][1]
        ocean_status = "Bad" if ocean_prob >= ocean_threshold else "Good"
 
        # ---------- 3. WHALE SPECTROGRAM MODEL ----------
        whale_confidence = 0.0
        whale_file       = None
 
        if 'spectrogram' in request.files:
            spectrogram_file = request.files['spectrogram']
            if spectrogram_file and spectrogram_file.filename:
                filename = secure_filename(spectrogram_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                spectrogram_file.save(filepath)
                whale_file = filename
 
                try:
                    img = Image.open(filepath).convert('RGB')
                    img = img.resize((128, 128))          # MATCHES TRAINING SIZE
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
 
                    whale_pred       = whale_model.predict(img_array, verbose=0)[0][0]
                    whale_confidence = float(whale_pred * 100)
                except Exception as e:
                    whale_confidence = 50.0
                    print(f"Whale prediction error: {e}")
 
        # Whale detected if confidence > 50% (sigmoid > 0.5 → class 1 = "whale")
        whale_detected = whale_confidence > 50.0
 
        # ---------- 4. AIS MODEL ----------
        # Map vessel_type string → numeric code your model expects
        vessel_type_map = {
            "cargo":   70.0,
            "tanker":  80.0,
            "fishing": 30.0,
            "tug":     52.0,
            "passenger": 60.0
        }
        vessel_code = vessel_type_map.get(vessel_type, 0.0)
 
        # Use realistic defaults for fields not in the form
        SOG     = float(request.form.get('SOG', 5.0))
        COG     = float(request.form.get('COG', 90.0))
        Heading = float(request.form.get('Heading', 90.0))
        Length  = float(request.form.get('Length', 100.0))
        Width   = float(request.form.get('Width', 20.0))
        Draft   = float(request.form.get('Draft', 5.0))
        LAT     = float(request.form.get('LAT', 13.0))
        LON     = float(request.form.get('LON', 80.0))
 
        SOG_kmh  = SOG * 1.852
        dist_km  = np.sqrt((LAT - 13.0)**2 + (LON - 80.0)**2)
 
        # Cargo & Status encoded → use 0 as default (unknown)
        cargo_enc  = 0
        status_enc = 0
 
        ais_input = np.array([[LAT, LON, SOG, COG, Heading,
                                Length, Width, Draft,
                                SOG_kmh, dist_km,
                                cargo_enc, status_enc]])
 
        ais_pred_enc    = ais_model.predict(ais_input)[0]
        ais_vessel_real = vessel_map.get(ais_pred_enc, vessel_code)
 
        # ---------- 5. MARINE MODEL ----------
        # Features: LiveTotal, DeadTotal, NewReddCount, CombinedReddCount, TotalFish
        LiveTotal        = float(request.form.get('LiveTotal', 80))
        DeadTotal        = float(request.form.get('DeadTotal', 5))
        NewReddCount     = float(request.form.get('NewReddCount', 3))
        CombinedReddCount= float(request.form.get('CombinedReddCount', 10))
        TotalFish        = LiveTotal + DeadTotal
 
        marine_input = np.array([[LiveTotal, DeadTotal,
                                   NewReddCount, CombinedReddCount, TotalFish]])
        marine_pred_enc = marine_model.predict(marine_input)[0]
        marine_status   = marine_le.inverse_transform([marine_pred_enc])[0]  # Low/Medium/High
 
        # ---------- 6. RISK SCORING (100-point scale) ----------
        score  = 0
        issues = []
 
        # Ocean (40 pts)
        if ocean_status == "Bad":
            score += 40
            issues.append({'name': 'Ocean Degradation',
                           'cause': f'pH:{pH:.1f}, O2:{oxygen:.1f}, Turbidity:{turbidity:.1f}'})
 
        # Whale (20 pts — whale present means zone is disturbed)
        if whale_detected:
            score += 20
            issues.append({'name': 'Whale Detected in Zone',
                           'cause': f'{whale_confidence:.1f}% confidence'})
 
        # Vessel (30 pts)
        vessel_risk_map = {'tanker': 30, 'cargo': 20, 'fishing': 15, 'tug': 10, 'passenger': 10}
        vessel_risk = vessel_risk_map.get(vessel_type, 5)
        score += vessel_risk
        issues.append({'name': 'Vessel Threat',
                       'cause': f'{vessel_type.title()} detected (AIS code: {ais_vessel_real})'})
 
        # Marine (10 pts)
        marine_risk = 0
        if marine_status == "High":
            marine_risk = 10
            issues.append({'name': 'Marine Ecosystem Stress',
                           'cause': 'High fish population decline'})
        elif marine_status == "Medium":
            marine_risk = 5
            issues.append({'name': 'Marine Ecosystem Moderate Stress',
                           'cause': 'Medium fish population decline'})
        score += marine_risk
 
        score = min(100, score)
 
        # ---------- 7. DECISION ----------
        if score < 30:
            decision        = "🟢 SAFE ZONE"
            risk_level      = "Low"
            endangered_prob = 12.5
            extinction_risk = "Minimal"
            action          = "Routine Monitoring"
        elif score < 60:
            decision        = "🟡 MODERATE RISK"
            risk_level      = "Medium"
            endangered_prob = 45.2
            extinction_risk = "Moderate"
            action          = "Enhanced Surveillance"
        else:
            decision        = "🔴 SILENT ZONE VIOLATION"
            risk_level      = "High"
            endangered_prob = 78.9
            extinction_risk = "Critical"
            action          = "IMMEDIATE INTERVENTION"
 
        # ---------- 8. STORE IN SESSION ----------
        session['results'] = {
            'pH': pH, 'oxygen': oxygen, 'turbidity': turbidity,
            'nitrate': nitrate, 'temperature': temperature,
            'vessel_type': vessel_type,
            'whale_confidence': round(whale_confidence, 2),
            'whale_detected': whale_detected,
            'ocean_status': ocean_status,
            'marine_status': marine_status,
            'ais_vessel': str(ais_vessel_real),
            'score': score,
            'decision': decision,
            'issues': issues,
            'risk_level': risk_level,
            'endangered_prob': endangered_prob,
            'extinction_risk': extinction_risk,
            'action': action,
            'whale_file': whale_file,
            'chart_data': {
                'labels': ['Ocean', 'Whale', 'Vessel', 'Marine'],
                'datasets': [
                    40 if ocean_status == 'Bad' else 0,
                    20 if whale_detected else 0,
                    vessel_risk,
                    marine_risk
                ]
            }
        }
 
        return redirect(url_for('results'))
 
    return render_template('upload.html')
 
# ============================================
# RESULTS
# ============================================
@app.route('/results')
def results():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('results.html', **session.get('results', {}))
 
# ============================================
# MAP DATA API
# ============================================
@app.route('/api/map-data')
def map_data():
    results = session.get('results', {})
    return jsonify({
        'vessels': [
            {'lat': 12.97, 'lng': 80.22, 'type': results.get('vessel_type','unknown'),
             'risk': results.get('score', 0)}
        ],
        'whales': [
            {'lat': 12.95, 'lng': 80.20,
             'confidence': results.get('whale_confidence', 0)}
        ],
        'risk_zones': [
            {'lat': 13.0, 'lng': 80.25, 'radius': 500,
             'score': results.get('score', 0)}
        ]
    })
 
# ============================================
# DOWNLOAD CSV REPORT
# ============================================
@app.route('/download_report')
def download_report():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
 
    results = session.get('results', {})
 
    report = {
        'pH':               results.get('pH'),
        'Dissolved_Oxygen': results.get('oxygen'),
        'Turbidity':        results.get('turbidity'),
        'Nitrate':          results.get('nitrate'),
        'Temperature':      results.get('temperature'),
        'Vessel_Type':      results.get('vessel_type'),
        'AIS_Vessel_Code':  results.get('ais_vessel'),
        'Whale_Confidence': results.get('whale_confidence'),
        'Whale_Detected':   results.get('whale_detected'),
        'Ocean_Status':     results.get('ocean_status'),
        'Marine_Status':    results.get('marine_status'),
        'Risk_Score':       results.get('score'),
        'Decision':         results.get('decision'),
        'Risk_Level':       results.get('risk_level'),
        'Endangered_Prob':  results.get('endangered_prob'),
        'Extinction_Risk':  results.get('extinction_risk'),
        'Action_Required':  results.get('action'),
    }
 
    df = pd.DataFrame([report])
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
 
    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='silent_zone_analysis_report.csv'
    )
 
# ============================================
# LOGOUT
# ============================================
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
 
# ============================================
# HOME
# ============================================
@app.route('/')
def home():
    return redirect(url_for('login'))
 
# ============================================
# RUN
# ============================================
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
