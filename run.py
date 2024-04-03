from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

from app import app
# Load data
df = pd.read_csv("/static/India Agriculture Crop Production.csv")
test = pd.read_csv("/static/India Agriculture Crop Production.csv")

# Preprocessing
df = df.dropna(subset=['Production', 'Crop'])
df = df.sort_values(by='Year', ascending=True)
df['Average Yield'] = df.groupby('Crop')['Yield'].transform('mean')
df['Recent_Yield_Increase'] = df.groupby(['Crop', 'District'])['Yield'].apply(lambda x: x.shift(-1) > x.shift(-5))
df.drop(['Production'], axis=1, inplace=True)

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()

df['State'] = le1.fit_transform(df['State'])
df['District'] = le2.fit_transform(df['District'])
df['Season'] = le3.fit_transform(df['Season'])
df['Crop'] = le4.fit_transform(df['Crop'])

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
df[['Area']] = scaler1.fit_transform(df[['Area']])
df[['Yield']] = scaler2.fit_transform(df[['Yield']])

df.drop(['Year', 'Production Units'], axis=1, inplace=True)

# Train model
X = df.drop(columns=['Yield'])
y = df['Yield']
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

# Helper functions
def get_crop_info(crop):
    crop_df = df[df['Crop'] == crop]
    average_yield = crop_df['Average Yield'].mean()
    recent_yield_increase = crop_df['Recent_Yield_Increase'].any()
    return average_yield, recent_yield_increase

# Flask routes
@app.route('/',methods = ['GET'])
def home():
    return "what man jayanth"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    state = data['State']
    district = data['District']
    season = data['Season']
    crop = data['Crop']
    area = data['Area']

    example_input = {
        'State': le1.transform([state])[0],
        'District': le2.transform([district])[0],
        'Season': le3.transform([season])[0],
        'Crop': le4.transform([crop])[0],
        'Area': scaler1.transform([[area]])[0][0],
        'Average Yield': get_crop_info(le4.transform([crop])[0])[0],
        'Recent_Yield_Increase': get_crop_info(le4.transform([crop])[0])[1]
    }

    prediction = model.predict([list(example_input.values())])
    inverse_prediction = scaler2.inverse_transform(prediction.reshape(-1,1))
    return jsonify({'yield_prediction': inverse_prediction[0][0]})



if __name__ == '__main__':
    app.run(debug=True)