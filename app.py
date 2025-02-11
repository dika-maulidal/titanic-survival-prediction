from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('model/titanic.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Pclass = int(request.form['Pclass'])
        Sex = 1 if request.form['Sex'] == 'male' else 0
        SibSp = int(request.form['SibSp'])
        Parch = int(request.form['Parch'])
        Fare = float(request.form['Fare'])
        Embarked = request.form['Embarked']

        Embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
        Embarked_encoded = Embarked_mapping.get(Embarked, 0)

        input_features = pd.DataFrame([[Pclass, Sex, SibSp, Parch, Fare, Embarked_encoded]],
                                      columns=['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked'])
        
        # Prediksi
        prediction = model.predict(input_features)
        result = 'Survived' if prediction[0] == 1 else 'Did not survive'

        return jsonify({'Prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
