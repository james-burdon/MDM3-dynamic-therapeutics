# import pyedflib
#
# path = r"D:\UOB\Year_3_UOB\mdm_hormone\OneDrive_2025-11-28\Data files\EDF\09-16-42.EDF"
#
# f = pyedflib.EdfReader(path)
#
# # Number of channels
# n_signals = f.signals_in_file
#
# # Channel names
# labels = f.getSignalLabels()
# print("Channels:", labels)
#
# # Read all signals
# signals = [f.readSignal(i) for i in range(n_signals)]
#
# # Example: print first 10 samples of channel 0
# print("First 10 samples of channel 0:", signals[0][:10])
#
# # Example: sampling rate of channel 0
# print("Sampling rate:", f.getSampleFrequency(0))
#
# f.close()

import pyedflib
import numpy as np

path = r"D:\UOB\Year_3_UOB\mdm_hormone\OneDrive_2025-11-28\Data files\EDF\09-16-42.EDF"

f = pyedflib.EdfReader(path)

n_channels = f.signals_in_file
labels = f.getSignalLabels()

signals = {}
sampling_rates = {}

for i, label in enumerate(labels):
    sig = f.readSignal(i)
    fs = f.getSampleFrequency(i)
    signals[label] = sig          # full channel signal
    sampling_rates[label] = fs    # sampling rate for that channel

f.close()

# Example: access ECG
ecg = signals["ECG"]
fs_ecg = sampling_rates["ECG"]

print("ECG length:", len(ecg), "fs:", fs_ecg)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(ecg[:5000])   # show first 5,000 samples (~20 seconds at 250Hz)
plt.title("ECG (first 20 seconds)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.show()

import matplotlib.pyplot as plt

ecg_data = raw.get_data(picks="ECG")[0]
fs = raw.info["sfreq"]
time = np.arange(len(ecg_data)) / fs

plt.figure(figsize=(12,4))
plt.plot(time[:5000], ecg_data[:5000])
plt.xlabel("Time (s)")
plt.ylabel("ECG amplitude")
plt.title("ECG: first 20 seconds")
plt.show()

