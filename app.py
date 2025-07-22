from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# ✅ Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form (in order!)
        features = [float(x) for x in request.form.values()]
        final_input = np.array([features])

        # Scale input
        scaled_input = scaler.transform(final_input)

        # Predict using the model
        prediction = model.predict(scaled_input)[0]

        # Format result
        if prediction == 1:
            result = "⚠️ The patient is at risk of death"
        else:
            result = "✅ The patient is not at immediate risk"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
