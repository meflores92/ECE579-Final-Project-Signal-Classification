import numpy as np
import torch
from PIL import Image, ImageOps
import matplotlib.cm as cm
from torchvision import transforms
import matplotlib.pyplot as plt

def preprocess_spectrogram(rxSpectrogram, target_size=(256, 256), colormap=cm.viridis):

    rxSpectrogram = rxSpectrogram[::-1]
    image = Image.fromarray(rxSpectrogram).convert('L')
    # Increase the contrast of the image
    image = ImageOps.autocontrast(image, cutoff=0.2)

    # Convert grayscale image to 3-channel image
    image = np.array(image)
    image = np.stack([image, image, image], axis=-1)

    # Resize the spectrogram
    image = Image.fromarray(image).resize(target_size, Image.Resampling.LANCZOS)
    mean = np.array([0.4934, 0.4934, 0.4934])
    std = np.array([0.1844, 0.1844, 0.1844])
    # Convert to PyTorch tensors
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)    

    plt.figure(figsize=(10,8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    # Apply the same transforms as during training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Apply transformations
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def save_spectrogram_as_rgb(rxSpectrogram, filename, target_size=(256, 256), colormap=cm.viridis):
    # Normalize the spectrogram to [0, 1]
    spectrogram_normalized = (rxSpectrogram - np.min(rxSpectrogram)) / (np.max(rxSpectrogram) - np.min(rxSpectrogram))

    # Apply colormap to convert to RGB
    spectrogram_color = colormap(spectrogram_normalized)[:, :, :3]  # Discard alpha channel

    # Convert to PIL Image for easy resizing
    spectrogram_image = Image.fromarray((spectrogram_color * 255).astype(np.uint8))

    # Resize the image
    spectrogram_resized = spectrogram_image.resize(target_size, Image.Resampling.LANCZOS)

    # Save the image
    spectrogram_resized.save(filename)

