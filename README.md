# Cadence-Peak-Detection-IMU
This project was a learning experience for signal processing and deriving conclusions from real world events.

Using acceleration data from one 9 sense IMU at my ankle. I gathered and detected cadence through a peak detection method referenced from the paper below. The initial data was collected using the PlatformIO extension. The acceleration norm was found first, the sampling rate was then brought down to 40hz, filtered through a low pass FIR filter, Transformed via a CWT, filtered one last time using a Savitzky–Golay filter, the peaks were extrapolated from the final result, cadence was calculated based off of each peak.

Prigent G, Barthelet E, Aminian K, Paraschiv-Ionescu A. Walking and running cadence estimation using a single trunk-fixed accelerometer for daily physical activities assessment. Annu Int Conf IEEE Eng Med Biol Soc. 2022 Jul;2022:3645-3648. doi: 10.1109/EMBC48229.2022.9871713. PMID: 36085794.
