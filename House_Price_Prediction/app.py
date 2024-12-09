from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
with open('C:/Users/NEHA/Downloads/House_Price_Prediction/neha/house_price_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form
        input_features = pd.DataFrame([{
            'beds': int(data['beds']),
            'baths': int(data['baths']),
            'size': int(data['size']),
            'lot_size': int(data['lot_size']),
            'zip_code': int(data['zip_code']),
            'size_units': 'sqft'
        }])

        # Make prediction
        prediction = model.predict(input_features)
        return render_template('index.html', prediction=f"${prediction[0]:,.2f}")

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
