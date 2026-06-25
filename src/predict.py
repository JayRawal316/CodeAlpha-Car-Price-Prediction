import joblib
import pandas as pd

# Load Model
model = joblib.load('models/car_price_model.pkl')

#Create a sample car with the same features used during training

sample_car = pd.DataFrame({
    'Present_Price' : [5.59],
    'Driven_kms' : [27000],
    'Owner' : [0],
    'Car_Age' : [8],
    'Fuel_Type_Diesel':[0],
    'Fuel_Type_Petrol':[1],
    'Selling_type_Individual':[1],
    'Transmission_Manual' : [1]
}
)

#Predict Selling Price -
predicted_price = model.predict(sample_car)
print(f"The predicted selling price of the car is: ₹ {predicted_price[0]:.2f} lakhs")