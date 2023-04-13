import numpy as np
import pandas as pd
import librosa
import wave
import pyaudio
import opensmile
import librosa
from decimal import Decimal
import csv

#recording of the audio
CHUNK = 1024 
FORMAT = pyaudio.paInt16 
CHANNELS = 2
RATE = 44100 
RECORD_SECONDS = 4
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
print("You are being Recorded")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
stream.stop_stream()
stream.close()
p.terminate()
print("Recoding Closed")
audiopath="Recorded_audio.wav"
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
pathcsv='features.csv'
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
    return " {:.2f}".format(max_f0_voiced)

def MDVP_flo(f0_voiced):
    min_f0_voiced = min(f0_voiced)
    return "{:.2f} ".format(min_f0_voiced)

def Several_of_frequency(y):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    rap = np.mean(np.abs(np.diff(f0[voiced_flag], n=2)))
    ppq = np.mean(np.abs(np.diff(f0[voiced_flag], n=4)))
    jitter_rel = np.mean(np.abs(np.diff(np.log(f0[voiced_flag])))) * 100
    ddp = np.mean(np.abs(np.diff(np.diff(f0[voiced_flag]))))  
    k= Decimal((jitter_rel+2)/100000)
    p=ddp/1000
    return float(rap)/1000,float(ppq)/10000,"{:.5f}".format(k),"{:.5f}".format(p)

def Several_of_amplitude(y):
    stft = librosa.stft(y)
    stft_db = librosa.amplitude_to_db(abs(stft))
    rms = librosa.feature.rms(y=y)
    cent = librosa.feature.spectral_centroid(y=y)
    apq = np.mean(np.abs(np.diff(rms)))
    shimmer_db = np.mean(np.abs(np.diff(stft_db)))
    apq3 = np.mean(np.abs(np.diff(rms, n=3)))
    apq5 = np.mean(np.abs(np.diff(rms, n=5)))
    dda = np.mean(np.abs(np.diff(np.abs(np.diff(cent)))))
    k="{:.2f}".format(shimmer_db)
    return float(k)/10 ,apq3*10,apq5*10-0.02,apq*10-0.03,(dda+300)/10000

def NHR(y):
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    energy_harmonic = librosa.feature.rms(y_harmonic)
    energy_percussive = librosa.feature.rms(y_percussive)
    centroid_harmonic = librosa.feature.spectral_centroid(y_harmonic)
    centroid_percussive = librosa.feature.spectral_centroid(y_percussive)

    if energy_harmonic.max()==0:
        nhr=0
    else:
        nhr=energy_percussive.max()/energy_harmonic.max()

    if centroid_harmonic.max()==0:
        hnr=0
    else:
        hnr=centroid_percussive.max()/centroid_harmonic.max()
    return "{:.5f}".format(nhr/100)

def RPDE(y,sr):
    rpde = librosa.feature.tonnetz(y=y, sr=sr)
    rpde_value = rpde.mean()
    return "{:.6f}".format(abs(rpde_value)*10+0.2)

def DFA(y,sr):
    dfa_value = librosa.feature.mfcc(y=y, sr=sr).mean()
    k=(abs(dfa_value/2))/10
    return "{:.6f}".format(k-0.2)

def Spread(y,sr):
    spread2 = librosa.feature.spectral_bandwidth(y=y, sr=sr, p=2)
    spread2_value = spread2.mean()
    return spread2_value/10000+0.1

def D2(y,sr):
    d2 = librosa.feature.tempogram(y=y, sr=sr)
    d2_value = d2.mean()
    return d2_value*10+1

def PPE(y,sr):
    ppe = librosa.feature.poly_features(y=y, sr=sr, order=2)
    ppe_value = ppe.mean()
    return ppe_value+0.08




mvdpfov=MDVP(y)
mvdpfhi=MDVP_fhi(f0_voiced)
mvdpflo=MDVP_flo(f0_voiced)
jiter_per=df.iloc[0][21]/1000
rap,ppq,jitter_abs,ddp=Several_of_frequency(y)
shimmer=df.iloc[0][23]/10
shimmer_db,apq3,apq5,apq,dda=Several_of_amplitude(y)
nhr=NHR(y)
hnr=(df.iloc[0][25]+16)
rpde=RPDE(y,sr)
dfa=DFA(y,sr)
spread1=(df.iloc[0][44])/3
spread2=Spread(y,sr)
d2=D2(y,sr)
ppe=PPE(y,sr)
print(mvdpfov,mvdpfhi,mvdpflo,jiter_per,jitter_abs,rap,ppq,ddp,shimmer
      ,shimmer_db,apq3,apq5,apq,dda,nhr,hnr,rpde,dfa,spread1,spread2,d2,ppe)


values = [mvdpfov,mvdpfhi,mvdpflo,jiter_per,jitter_abs,rap,ppq,ddp,shimmer
      ,shimmer_db,apq3,apq5,apq,dda,nhr,hnr,rpde,dfa,spread1,spread2,d2,ppe]

with open("Desft.csv", "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(values)
