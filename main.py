from flask import Flask, render_template, request, jsonify, session
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-this-in-production'

# Load the saved models
with open('tuned_dtr.pkl', 'rb') as file:
    dtr_model = pickle.load(file)

with open('tuned_rfr.pkl', 'rb') as file:
    rfr_model = pickle.load(file)


airline_dict = {
    'Air Asia': 0,
    'Air India': 1,
    'GoAir': 2,
    'IndiGo': 3,
    'Jet Airways': 4,
    'Jet Airways Business': 5,
    'Multiple carriers': 6,
    'Multiple carriers Premium economy': 7,
    'SpiceJet': 8,
    'Trujet': 9,
    'Vistara': 10,
    'Vistara Premium economy': 11
}


destination_dict = {
    'Kolkata': 0,
    'Hyderabad': 1,
    'Delhi': 2,
    'Banglore': 3,
    'Cochin': 4
}

# Source cities
source_cities = ['Banglore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai']

# Estimated flight times (in minutes) between cities
flight_times = {
    ('Banglore', 'Delhi'): 165,
    ('Banglore', 'Kolkata'): 165,
    ('Banglore', 'Hyderabad'): 75,
    ('Banglore', 'Cochin'): 60,
    ('Chennai', 'Delhi'): 165,
    ('Chennai', 'Kolkata'): 150,
    ('Chennai', 'Hyderabad'): 75,
    ('Chennai', 'Cochin'): 75,
    ('Chennai', 'Banglore'): 50,
    ('Delhi', 'Banglore'): 165,
    ('Delhi', 'Kolkata'): 135,
    ('Delhi', 'Hyderabad'): 135,
    ('Delhi', 'Cochin'): 210,
    ('Kolkata', 'Delhi'): 135,
    ('Kolkata', 'Banglore'): 165,
    ('Kolkata', 'Hyderabad'): 120,
    ('Kolkata', 'Cochin'): 180,
    ('Mumbai', 'Delhi'): 135,
    ('Mumbai', 'Kolkata'): 165,
    ('Mumbai', 'Hyderabad'): 90,
    ('Mumbai', 'Banglore'): 90,
    ('Mumbai', 'Cochin'): 120,
}

@app.route('/')
def index():
    # Get last form data from session if available
    last_data = session.get('last_prediction', {})
    return render_template('index.html', 
                         airlines=list(airline_dict.keys()),
                         sources=source_cities,
                         destinations=list(destination_dict.keys()),
                         last_data=last_data)

@app.route('/get_flight_time', methods=['POST'])
def get_flight_time():
    data = request.json
    source = data.get('source')
    destination = data.get('destination')
    
    # Get estimated flight time
    flight_time = flight_times.get((source, destination), 120)
    hours = flight_time // 60
    minutes = flight_time % 60
    
    return jsonify({
        'flight_time': flight_time,
        'formatted_time': f"{hours}h {minutes}m"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        airline = request.form['airline']
        source = request.form['source']
        destination = request.form['destination']
        journey_day = int(request.form['journey_day'])
        journey_month = int(request.form['journey_month'])
        departure_hour = int(request.form['departure_hour'])
        departure_min = int(request.form['departure_min'])
        arrival_hour = int(request.form['arrival_hour'])
        arrival_min = int(request.form['arrival_min'])
        total_stops = int(request.form['total_stops'])
        
        # Store form data in session for later use
        session['last_prediction'] = {
            'airline': airline,
            'source': source,
            'destination': destination,
            'journey_day': journey_day,
            'journey_month': journey_month,
            'departure_hour': departure_hour,
            'departure_min': departure_min,
            'arrival_hour': arrival_hour,
            'arrival_min': arrival_min,
            'total_stops': total_stops
        }
        
        # Calculate total minutes
        total_min = (arrival_hour - departure_hour) * 60 + (arrival_min - departure_min)
        if total_min < 0:
            total_min += 24 * 60  # Handle overnight flights
        
        # Encode source (one-hot encoding)
        source_encoded = {city: 0 for city in source_cities}
        if source in source_encoded:
            source_encoded[source] = 1
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Airline': [airline_dict.get(airline, 0)],
            'Destination': [destination_dict.get(destination, 0)],
            'Total_Stops': [total_stops],
            'Journey_day': [journey_day],
            'Journey_month': [journey_month],
            'Arrival_hour': [arrival_hour],
            'Arrival_min': [arrival_min],
            'Departure_hour': [departure_hour],
            'Departure_min': [departure_min],
            'Total_min': [total_min],
            'Source_Banglore': [source_encoded['Banglore']],
            'Source_Chennai': [source_encoded['Chennai']],
            'Source_Delhi': [source_encoded['Delhi']],
            'Source_Kolkata': [source_encoded['Kolkata']],
            'Source_Mumbai': [source_encoded['Mumbai']]
        })
        
        # Make predictions with both models
        rf_prediction = rfr_model.predict(input_data)[0]
        dt_prediction = dtr_model.predict(input_data)[0]
        
        # Calculate average prediction
        avg_prediction = (rf_prediction + dt_prediction) / 2
        
        return render_template('result.html',
                             rf_prediction=round(rf_prediction, 2),
                             dt_prediction=round(dt_prediction, 2),
                             avg_prediction=round(avg_prediction, 2),
                             airline=airline,
                             source=source,
                             destination=destination,
                             journey_day=journey_day,
                             journey_month=journey_month,
                             departure_time=f"{departure_hour:02d}:{departure_min:02d}",
                             arrival_time=f"{arrival_hour:02d}:{arrival_min:02d}",
                             stops=total_stops)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
