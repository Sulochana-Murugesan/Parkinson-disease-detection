import numpy as np
import pandas as pd
import librosa
import pyaudio
import wave
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score



parkinsons_data = pd.read_csv('Parkinsson disease.csv')
parkinsons_data.groupby('status').mean()
X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

input_data = (116.682,131.111,111.555,0.0105,0.00009,0.00544,0.00781,0.01633,0.05233,0.482,0.02757,0.03858,0.0359,0.0827,0.01309,20.651	,0.429895,0.825288,-4.443179,0.311173,2.342259,0.332634)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)
prediction = model.predict(std_data)
print(prediction)


if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons")