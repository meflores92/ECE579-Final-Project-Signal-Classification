import adi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plot_helpers as ph
import helper_spec_sense_spectrogram_image as image
import torch
import matplotlib.patches as mpatches
from model_definition import DeepLabV3SqueezeNet, UNet_V2
# Configuration parameters
num_classes = 3
model = UNet_V2().to('cpu')
model.load_state_dict(torch.load('C:/Users/meflo/Downloads/pruned_quantized_4_bit_model.pth', map_location=torch.device('cpu')))
model.to('cpu')
model.eval()

model.eval()
classNames = ["Noise" "NR" "LTE"]
fc = 2355e6  # Center frequency
fs = 61.44e6    # Sample rate
frameDuration = 40 * 1e-3
buff_size = 1024 * (1 + int(frameDuration * fs) // 1024)
Nfft = 4096
overlap = 10
imageSize = (256,256)
meanAllScores = np.zeros((imageSize[0], imageSize[1], num_classes))
segResults = np.zeros((imageSize[0], imageSize[1], 10))
# Create a PlutoSDR device instance
sdr = adi.Pluto(uri="ip:192.168.2.1")  # Replace with your PlutoSDR's URI

# Configure device parameters
sdr.sample_rate = int(fs)
sdr.rx_lo = int(fc)
sdr.rx_buffer_size = buff_size

def visualize_classes(rxSpectrogram, fc, fs, x, output_classes):
    # Create a color map for visualization
    color_map = np.array([[0, 0, 0],    # Black for Noise
                          [0, 255, 0],  # Green for NR
                          [0, 0, 255]]) # Blue for LTE

    # Map the class output to RGB colors

    output_classes = output_classes[:,:,0]
    rgb_image = color_map[output_classes]

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Spectrogram
    spect_img = axs[0].imshow(rxSpectrogram[::-1], aspect='auto', extent=[(fc-fs/2)/1e6, (fc+fs/2)/1e6, 0, len(x)/fs])
    axs[0].set_title('Spectrogram')
    axs[0].set_xlabel('Frequency (MHz)')
    axs[0].set_ylabel('Time (s)')
    fig.colorbar(spect_img, ax=axs[0], label='Intensity')

    # RGB Image
    axs[1].imshow(rgb_image)
    axs[1].set_title('Class Predictions')
    axs[1].axis('off')

    # Create a legend for the RGB image
    classes = ['Noise', 'NR', 'LTE']
    colors = [(0, 0, 0), (0, 1, 0), (0, 0, 1)]  # Corresponding colors in RGB
    patches = [mpatches.Patch(color=colors[i], label=classes[i]) for i in range(len(classes))]
    axs[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

# Collect and process data
for frameCnt in range(1):
    # Receive samples
    rxWave = sdr.rx()
    
    # Process the received samples
    rxSpectrogram, x = ph.spectrogram(rxWave, fc, fs)
    print(rxSpectrogram)
    # Convert to tensor and add batch dimension
    preprocessed_spectrogram = image.preprocess_spectrogram(rxSpectrogram, target_size=(256, 256))#, colormap=cm.viridis)
    # Now, rxSpectrogram is in the format [1, 1, 256, 256]
    print(preprocessed_spectrogram)
    # segResult = model(rxSpectrogram)#torch.tensor(rxSpectrogram.copy(), dtype=torch.float32))
    with torch.no_grad():
        predictions = model(preprocessed_spectrogram)

    # Process the predictions
    print(predictions)
    _, predictions = torch.max(predictions.data, 1)
    # if predictions.dim() == 4:
        # predictions = predictions[0]  # Select the first image from the batch
    # Normalize the tensor to [0, 1]
    # min_val = torch.min(predictions)
    # max_val = torch.max(predictions)
    # predictions = (predictions - min_val) / (max_val - min_val)
    print(predictions)
    # Convert to numpy and change layout to (Height, Width, Channel)
    output_array = predictions.numpy()
    output_array = (output_array * 255).astype(np.uint8)
    output_array = np.transpose(output_array, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)

    output_array = np.round(output_array / 127.5) * 127.5
    print(output_array)
    # Map to class labels
    output_classes = np.zeros_like(output_array, dtype=np.uint8)
    output_classes[output_array == 0] = 0     # Noise
    output_classes[output_array == 127] = 1   # NR
    output_classes[output_array == 255] = 2   # LTE

    print(output_classes)
    visualize_classes(rxSpectrogram, fc, fs, x, output_classes)

