
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.signal import find_peaks
from scipy.fft import fft
from scipy.signal import welch
from scipy.stats import gaussian_kde
def extract_eeg_features(X_eeg):
    n_samples = X_eeg.shape[0]
    n_channels = X_eeg.shape[1]
    features = []
    for i in range(n_samples):
        eeg_sample = X_eeg[i]
        channel_features = []

        for j in range(n_channels):
            freqs, psd = welch(eeg_sample[j], fs=128)  

            delta_power = np.sum(psd[(freqs >1) & (freqs <=3)])
            theta_power = np.sum(psd[(freqs >=3) & (freqs < 8)])
            alpha_power = np.sum(psd[(freqs >=8) & (freqs < 12)])
            beta_power = np.sum(psd[(freqs >=12) & (freqs<30)])
            Gamma__power= np.sum(psd[(freqs >=25)])
            total_power = delta_power + theta_power + alpha_power + beta_power+Gamma__power
            probabilities = [p / total_power for p in [delta_power, theta_power, alpha_power, beta_power,Gamma__power]]
            entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])

            mean = np.mean(eeg_sample[j])
            std = np.std(eeg_sample[j])

            de = differential_entropy(eeg_sample[j])

            channel_features.append([delta_power, theta_power, alpha_power, beta_power,Gamma__power, entropy, de])

        features.append(np.concatenate(channel_features))

    features = np.array(features)
    return features
def extract_ecg_features(X_ecg):
    n_samples = X_ecg.shape[0]
    n_features = 4
    features = []

    for i in range(n_samples):
        X_ecg_i = X_ecg[i].reshape(-1)  

        peaks, _ = find_peaks(X_ecg_i, height=0.3, distance=20, prominence=0.25)  
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) 
            rr_intervals = rr_intervals[(rr_intervals > 200) & (rr_intervals < 2000)]  
            if len(rr_intervals) > 0:
                mean_rr = np.mean(rr_intervals)
                std_rr = np.std(rr_intervals)
                rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) 
                sdnn = np.std(rr_intervals)  

                N = len(X_ecg_i)
                fft_vals = fft(X_ecg_i)
                fft_vals = np.abs(fft_vals[:N // 2]) 
                freqs = np.fft.fftfreq(N, 1 / 500)[:N // 2]

                lf_band = (0.04, 0.15) 
                hf_band = (0.15, 0.4)  
                lf_power = np.sum(fft_vals[(freqs >= lf_band[0]) & (freqs <= lf_band[1])])
                hf_power = np.sum(fft_vals[(freqs >= hf_band[0]) & (freqs <= hf_band[1])])

                lf_hf_ratio = lf_power / (hf_power + 1e-6)  
                total_power = np.sum(fft_vals)  

                features_i = np.array([
                  #  mean_rr, std_rr,
                    #skew_rr, kurt_rr, , total_power
                    rmssd,  sdnn, lf_power, hf_power,lf_hf_ratio
                ])
            else:
                features_i = np.zeros(n_features)
        else:
            features_i = np.zeros(n_features)

        features.append(features_i)

    features = np.array(features)
    return features
def differential_entropy(signal):
    kde = gaussian_kde(signal)
    log_prob_density = np.log(kde(signal))
    de = np.mean(log_prob_density) * -1  
    return de

def add_gaussian_noise(data, mean=8, std=8):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

def average_features(X_eeg_features):
    n_samples = X_eeg_features.shape[0]
    n_features_per_channel = 8
    n_channels = 14

    averaged_features = np.zeros((n_samples, n_features_per_channel))

    for i in range(n_features_per_channel):
        channel_feature = X_eeg_features[:, i::n_features_per_channel]
        averaged_features[:, i] = np.mean(channel_feature, axis=1)

    return averaged_features
data_path = "C:/Users/cyl/Desktop/naodian/001.npz"
with np.load(data_path) as npzfile:
    X = npzfile['X']
    y = npzfile['y']

mu_X = X.mean(axis=0)
sigma_X = X.std(axis=0)
X = (X - mu_X) / sigma_X
Y=y
X_eeg=X
X_eeg=extract_eeg_features(X_eeg)
X_eeg=average_features(X_eeg)
df_ecg = pd.read_csv(r'C:\Users\cyl\Desktop\xindian\data\001.csv')
ecg_data = df_ecg['ECG_Signal'].values

time_series_length = 128
step_size = 32
X_ecg = [ecg_data[i:i + time_series_length] for i in range(0, len(ecg_data) - time_series_length + 1, step_size)]
X_ecg = np.array(X_ecg)
X_ecg = extract_ecg_features(X_ecg)
scaler = StandardScaler()

X_ecg_scaled = scaler.fit_transform(X_ecg)*100       
X_ecg_pca =X_ecg_scaled*100

y=Y
def add_gaussian_noise(data, mean=0.0000001, std=0.0000001):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise
X_ecg=add_gaussian_noise((X_ecg_pca))
min_len = min(X_eeg.shape[0], X_ecg.shape[0])
X_eeg = X_eeg[:min_len]
X_ecg = X_ecg[:min_len]
X_eeg_ecg = np.concatenate((X_eeg,X_ecg), axis=1)
min_len = min(X_eeg_ecg.shape[0], y.shape[0])
X_eeg_ecg = X_eeg_ecg[:min_len]
y = y[:min_len]
print("X_eeg_ecg.shape",X_eeg_ecg.shape)
X_train, X_validate, Y_train, Y_validate = train_test_split(X_eeg_ecg,y, test_size=0.3, random_state=42)






