# Image Steganalysis with EfficientNet

<!-- ABOUT THE PROJECT -->
## About The Project

<br/>
Images may contain hidden messages that aren’t part of its regular contents. The same technology employed for digital watermarking is also misused by crime rings. Law enforcement must now use steganalysis to detect these messages as part of their investigations.

The objective of this project is to determine if a given image contains hidden messages.
<br/>

<p align="center">
  <img src="/image/image.png" alt="Competition image" width="800" height="500"/>
</p>


This is my solution to the [ALASKA2 Image Steganalysis](https://www.kaggle.com/c/alaska2-image-steganalysis) Kaggle competition. I used an EfficientNet-B2 to train on the image datasets that was split according to their quality factors

Result: Weighted AUC score of 0.907. Ranked 209 out of 1095 teams in the [private leaderboard](https://www.kaggle.com/c/alaska2-image-steganalysis/leaderboard).
<br/><br/>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running, follow these simple example steps.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/stephenllh/steganalysis-alaska2.git
   ```

1. Change directory
   ```sh
   cd steganalysis-alaska2
   ```

2. Install packages
   ```sh
   pip install requirements.txt
   ```

<br/>

<!-- USAGE EXAMPLES -->
## Usage

1. Enter the directory called `input`
   ```sh
   cd input
   ```

2. Download the image dataset into the folder called `input`
    - Option 1: Use Kaggle API
      - `pip install kaggle`
      - `kaggle competitions download -c alaska2-image-steganalysis`
    - Option 2: Download the dataset from the [competition website](https://www.kaggle.com/c/alaska2-image-steganalysis/data).

3. Run the training script
   ```sh
   cd ..
   python train.py
   ```

4. (Optional) Run the inference script
   ```sh
   python inference.py
   ```

<br/>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.
<br></br>


<!-- CONTACT -->
## Contact

Stephen Lau - [Email](stephenlaulh@gmail.com) - [Twitter](https://twitter.com/StephenLLH) - [Kaggle](https://www.kaggle.com/faraksuli)


