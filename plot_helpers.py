import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt


#-----------------------------------------------------------------------------------------

def plotme(thing, name="", show_grid = False, show_pips = False):
	#return #use as global plotting disable
	plt.plot(thing)
	if show_pips: plt.plot(thing, "bo")
	plt.title(name)
	if show_grid: plt.grid()
	plt.show()


#-----------------------------------------------------------------------------------------

def spectrogram(x, rx_center, rx_sample_rate):
	#fft_size and row_offset must be powers of two, and row offset should be no larger than fft_size
	fft_size = 2**10
	row_offset = 2**5 #added to make the spectrogram smoother
	
	if len(x) < fft_size:
		raise ValueError('x is too short for FFT')
		
	# Calculate num_rows with a check to avoid division by zero
	row_calculation = (row_offset * len(x) / fft_size) - row_offset
	if row_calculation <= 0:
		raise ValueError("Invalid calculation for num_rows. Check fft_size and row_offset settings.")
        
	num_rows = int(np.floor(row_offset*len(x)/fft_size))-row_offset #fixes overrunning the end of the data array
	print("number of spectrogram rows: %s" % num_rows)
	spectrogram = np.zeros((num_rows, fft_size), dtype=np.int8)
	for i in range(num_rows):
		start_idx = i*fft_size//row_offset
		stop_idx = start_idx + fft_size
		segment = x[start_idx:stop_idx]
		if len(segment) != fft_size:
			raise ValueError("Segment length is not equal to fft_size.")
		#spectrogram[i,:] = [ int(x) if x is not np.inf else 0 for x in 10*np.log10(np.abs(fftshift(fft(x[start_idx:stop_idx])))**2) ]
		test = 10*np.log10((np.abs(fftshift(fft(x[start_idx:stop_idx])))**2) + 1e-10)
		test = [int(x) if x != float('inf') and x != float('-inf') else 0 for x in test]
		spectrogram[i,:] = test
		if not i%100: print(i)
	
	#plt.imshow(spectrogram[::-1], aspect='auto', extent = [(rx_center-rx_sample_rate/2)/1e6, (rx_center+rx_sample_rate/2)/1e6, 0, len(x)/rx_sample_rate])
	#plt.xlabel("Frequency [MHz]")
	#plt.ylabel("Time [s]")
	#plt.show()
	return spectrogram, x

#-----------------------------------------------------------------------------------------
