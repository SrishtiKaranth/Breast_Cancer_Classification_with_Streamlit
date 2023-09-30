import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle



#loading saved model
loaded_model = pickle.load(open("C:\\Users\\srish\\Desktop\\ML Projects\\Breast Cancer Detection\\trained_model.sav", 'rb'))
loaded_scaler = pickle.load(open("C:\\Users\\srish\\Desktop\\ML Projects\\Breast Cancer Detection\\scaler.pkl", 'rb'))



single_obs=[[17.99,10.36,120.80,999.0,0.11840,0.27760,0.3001,0.14710,0.2419,0.07818,1.095,0.9052,8.567,153.4,0.00639,0.0494,0.0536,0.0158,0.0303,0.0016,23.38,17.33,184.6,2019,0.1622,0.665,0.7119,0.2654,0.4601,0.1189]]
standardized_input = loaded_scaler.transform(single_obs)
prediction = loaded_model.predict(standardized_input)
if(prediction==1):
  print("Cancer is Malignant")
else:
  print("Cancer is Bening")