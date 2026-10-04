from flask import Flask, render_template, request, jsonify, session
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'student_predictor_2024'

# ── Load bundle ────────────────────────────────────────────────────────────────
with open('student_bundle.pkl', 'rb') as f:
    bundle = pickle.load(f)

models        = bundle['models']
best_model    = bundle['best_model']
encoders      = bundle['encoders']
feature_names = bundle['feature_names']
model_results = bundle['results']
tuning_params = bundle.get('tuning_params', {})

# Dataset averages (from real data) — used for SHAP-style explanation
DATASET_MEANS = {
    'gender': 0.482,
    'race_ethnicity': 2.29,
    'parental_level_of_education': 2.91,
    'lunch': 0.645,
    'test_preparation_course': 0.358,
    'reading_score': 69.17,
    'writing_score': 68.05,
}
MEAN_MATH = 66.09

FEATURE_LABELS = {
    'gender':                       'Gender',
    'race_ethnicity':               'Race / Ethnicity',
    'parental_level_of_education':  'Parental Education',
    'lunch':                        'Lunch Type',
    'test_preparation_course':      'Test Prep Course',
    'reading_score':                'Reading Score',
    'writing_score':                'Writing Score',
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def encode_input(form):
    row = [
        encoders['gender'].transform([form['gender']])[0],
        encoders['race_ethnicity'].transform([form['race_ethnicity']])[0],
        encoders['parental_level_of_education'].transform([form['parental_level_of_education']])[0],
        encoders['lunch'].transform([form['lunch']])[0],
        encoders['test_preparation_course'].transform([form['test_preparation_course']])[0],
        float(form['reading_score']),
        float(form['writing_score']),
    ]
    return pd.DataFrame([row], columns=feature_names)

def grade_from_score(score):
    if score >= 90: return 'A', '#10b981'
    if score >= 80: return 'B', '#6366f1'
    if score >= 70: return 'C', '#f59e0b'
    if score >= 60: return 'D', '#f97316'
    return 'F', '#ef4444'

def confidence_range(score, model_name):
    rmse = model_results[model_name]['RMSE']
    return max(0, round(score - rmse)), min(100, round(score + rmse))

def compute_shap_contributions(X_row, model_name):
    """
    Feature contributions relative to dataset mean prediction.
    Linear models: coef * (value - mean_value)
    Tree models:   importance-weighted scaled to prediction gap
    """
    model = models[model_name]
    X_mean = pd.DataFrame([DATASET_MEANS], columns=feature_names)

    pred_student = float(model.predict(X_row)[0])
    pred_mean    = float(model.predict(X_mean)[0])
    total_gap    = pred_student - pred_mean

    if hasattr(model, 'coef_'):
        coefs = model.coef_
        contributions = [
            coefs[i] * (float(X_row.iloc[0, i]) - DATASET_MEANS[feat])
            for i, feat in enumerate(feature_names)
        ]
    else:
        importances = model.feature_importances_
        raw = [
            importances[i] * (float(X_row.iloc[0, i]) - DATASET_MEANS[feat])
            for i, feat in enumerate(feature_names)
        ]
        raw_sum = sum(abs(r) for r in raw) or 1
        contributions = [r / raw_sum * total_gap for r in raw]

    result = []
    for i, feat in enumerate(feature_names):
        c = contributions[i]
        result.append({
            'feature':      feat,
            'label':        FEATURE_LABELS[feat],
            'value':        round(float(X_row.iloc[0, i]), 1),
            'contribution': round(c, 2),
            'direction':    'positive' if c >= 0 else 'negative',
            'abs':          abs(round(c, 2)),
        })

    result.sort(key=lambda x: x['abs'], reverse=True)
    return result, round(pred_student, 2), round(pred_mean, 2)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html',
                           model_results=model_results,
                           best_model=best_model,
                           tuning_params=tuning_params,
                           history=session.get('history', [])[-5:][::-1])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        selected_model = request.form.get('selected_model', best_model)
        X = encode_input(request.form)

        prediction = float(models[selected_model].predict(X)[0])
        prediction = round(max(0, min(100, prediction)), 1)
        grade, grade_color = grade_from_score(prediction)
        low, high = confidence_range(prediction, selected_model)
        r2 = model_results[selected_model]['R2']

        shap_contribs, pred_val, mean_val = compute_shap_contributions(X, selected_model)

        history = session.get('history', [])
        history.append({
            'model':      selected_model,
            'prediction': prediction,
            'grade':      grade,
            'reading':    request.form['reading_score'],
            'writing':    request.form['writing_score'],
            'prep':       request.form['test_preparation_course'],
        })
        session['history'] = history[-10:]

        return render_template('index.html',
                               model_results=model_results,
                               best_model=best_model,
                               tuning_params=tuning_params,
                               prediction=prediction,
                               grade=grade,
                               grade_color=grade_color,
                               low=low, high=high,
                               selected_model=selected_model,
                               r2=r2,
                               shap_contribs=shap_contribs,
                               mean_math=MEAN_MATH,
                               history=session['history'][-5:][::-1],
                               form=request.form)
    except Exception as e:
        return render_template('index.html',
                               model_results=model_results,
                               best_model=best_model,
                               tuning_params=tuning_params,
                               error=str(e),
                               history=session.get('history', [])[-5:][::-1])

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html',
                           model_results=model_results,
                           best_model=best_model,
                           tuning_params=tuning_params)

@app.route('/clear_history')
def clear_history():
    session.pop('history', None)
    return ('', 204)

# ── REST API ───────────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    REST API — POST JSON to get prediction + explanation.

    Example:
      curl -X POST http://localhost:5000/api/predict \
        -H "Content-Type: application/json" \
        -d '{
          "gender": "female",
          "race_ethnicity": "group C",
          "parental_level_of_education": "bachelor s degree",
          "lunch": "standard",
          "test_preparation_course": "completed",
          "reading_score": 75,
          "writing_score": 80
        }'
    """
    try:
        data = request.get_json(force=True)
        required = ['gender','race_ethnicity','parental_level_of_education',
                    'lunch','test_preparation_course','reading_score','writing_score']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({'success': False, 'error': f'Missing fields: {missing}'}), 400

        selected_model = data.get('model', best_model)
        if selected_model not in models:
            return jsonify({'success': False,
                            'error': f'Unknown model. Options: {list(models.keys())}'}), 400

        X = encode_input(data)
        prediction = float(models[selected_model].predict(X)[0])
        prediction = round(max(0, min(100, prediction)), 1)
        grade, _   = grade_from_score(prediction)
        low, high  = confidence_range(prediction, selected_model)

        shap_contribs, pred_val, mean_val = compute_shap_contributions(X, selected_model)

        return jsonify({
            'success':       True,
            'prediction':    prediction,
            'grade':         grade,
            'confidence':    {'low': low, 'high': high},
            'model_used':    selected_model,
            'model_r2':      model_results[selected_model]['R2'],
            'model_rmse':    model_results[selected_model]['RMSE'],
            'baseline_mean': mean_val,
            'explanation':   [
                {
                    'feature':      c['label'],
                    'contribution': c['contribution'],
                    'direction':    c['direction'],
                }
                for c in shap_contribs[:5]
            ],
            'all_models': {
                name: {'R2': res['R2'], 'RMSE': res['RMSE']}
                for name, res in model_results.items()
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def api_models():
    """Returns all models, metrics, and best model name."""
    return jsonify({
        'best_model':    best_model,
        'models':        model_results,
        'tuning_params': tuning_params,
        'features':      feature_names,
    })

@app.route('/api/docs')
def api_docs():
    """Interactive API documentation page."""
    return render_template('api_docs.html',
                           model_results=model_results,
                           best_model=best_model)

if __name__ == '__main__':
    app.run(debug=True)
