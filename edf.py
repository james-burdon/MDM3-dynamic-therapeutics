import mne

raw = mne.io.read_raw_edf(r"C:\Users\jburd\Desktop\MDM3-dynamic-therapeutics\09-16-42.EDF", preload=True)


# Quick summary
print(raw.info)

# Plot first 10 channels over 10 seconds
raw.plot(duration=10, n_channels=10, show=True)
