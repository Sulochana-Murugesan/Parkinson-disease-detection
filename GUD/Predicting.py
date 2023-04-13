import numpy as np
import pandas as pd
import librosa
import wave
import pyaudio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import opensmile
import librosa


#model building 
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



#recording of the audio
CHUNK = 1024 
FORMAT = pyaudio.paInt16 
CHANNELS = 2
RATE = 44100 
RECORD_SECONDS = 5 
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
print("you are being recorded")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
stream.stop_stream()
stream.close()
p.terminate()
print("recoding closed")
audiopath="test2.wav"
wf = wave.open(audiopath, "wb")
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b"".join(frames))
wf.close()
print(audiopath)

##Featature extraction
smile = opensmile.Smile(feature_set=opensmile.FeatureSet.GeMAPSv01b,feature_level=opensmile.FeatureLevel.Functionals)
# extract features from an audio file
features = smile.process_file(audiopath)
pathcsv='feas.csv'
features.to_csv(pathcsv, index=False)
df = pd.read_csv(pathcsv)

#extraction of the missing features
y, sr = librosa.load(audiopath)
stft = librosa.stft(y)
stft_db = librosa.amplitude_to_db(abs(stft))
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
f0_voiced = f0[voiced_flag > 0]


def MDVP(y):
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    f0, voiced_flag, voiced_probs = librosa.pyin(y_harmonic, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    voiced_f0 = f0[voiced_flag > 0]
    mean_f0 = librosa.hz_to_midi(voiced_f0[voiced_f0 > 0]).mean()
    return "{:.2f}".format(librosa.midi_to_hz(mean_f0))

def MDVP_fhi(f0_voiced):
    max_f0_voiced = max(f0_voiced)
    print( " {:.2f}".format(max_f0_voiced))

def MDVP_flo(f0_voiced):
    min_f0_voiced = min(f0_voiced)
    print( "{:.2f} ".format(min_f0_voiced))

def Several_of_frequency(y):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    rap = np.mean(np.abs(np.diff(f0[voiced_flag], n=2)))
    ppq = np.mean(np.abs(np.diff(f0[voiced_flag], n=4)))
    return float(rap)/1000,float(ppq)/10000

def DFA(y,sr):
    dfa_value = librosa.feature.mfcc(y=y, sr=sr).mean()
    (abs(dfa_value)+40)/1000


mvdpfov=MDVP(y)
mvdpfhi=MDVP_fhi(f0_voiced)
mvdpflo=MDVP_flo(f0_voiced)
rap,ppq=Several_of_frequency(y)
jiter_per=df.iloc[0][21]
shimmer=df.iloc[0][23]
hnr=df.iloc[0][25]
spread1=df.iloc[0][45]

# Load CSV file, extract only "col1" and "col2" columns
df = pd.read_csv(pathcsv, usecols=[])

# Display the extracted data
print(df)



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



