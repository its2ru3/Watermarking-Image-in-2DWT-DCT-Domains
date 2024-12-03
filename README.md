## Blind Image Watermarking in DWT-DCT Domains

This project implements a blind image watermarking algorithm in the Discrete Wavelet Transform (DWT) and Discrete Cosine Transform (DCT) domains. The watermark is embedded into the image using block DCT on the LL1 component of the image, ensuring that the watermark remains protected.

The **report** for this project can be found [here](https://drive.google.com/file/d/1QJBaFD2Gk-eMFxo17dVSMVNumTm-0VDA/view?usp=drive_link).

### Features

- **Watermark Embedding & Extraction**: Perform watermark embedding and extraction in the DWT-DCT domains.
- **JPEG Compression-like Methodology**: Similar to JPEG compression after extracting the LL1 component of the image.
- **QR Code Watermark**: The algorithm embeds a QR code that can be scanned even after the image undergoes heavy attacks.
- **Robustness**: The watermark is robust against common image manipulations such as noise, compression, and cropping.

### Installation

To get started with this project, follow the steps below:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/its2ru3/Watermarking-Image-in-DWT-DCT-Domains.git
   ```

    Install Dependencies:
  ``` bash
    pip install -r requirements.txt
   ```

### Usage

  Place your image and watermark (QR code) in the dataset directory.

  Modify the global variables image and watermark in the main.py file to reference your test image and watermark QR code.

  Run the main.py file in your preferred code editor. Upon running, a new folder (named after the image file) will be created in the dataset folder containing: Watermarked image, Decoded watermark, Attacked image, etc.

  To Test Robustness:
        Choose any of the available attacks from the files in the utils folder.
        Replace the line Y_atk=jpeg_compression(Y_w, 20) with your preferred attack.

### License

This project is licensed under the MIT License.
