# NREIP
*Fall 2025 NREIP Internship Project under the guidance of Dr. Aaron Cohen.

OBJECTIVE:
This is the code used to create a machine learning script in PyTorch to identify audio snips into their respective classes. Model works with CPU and CUDA if available.

DATA CLASSIFICATON:
On the csv there are labels 0-6, these correspond to different classes of audios:
0 - Communication
1 - Gunshot
2 - Footsteps
3 - Shelling
4 - Vehicle
5 - Helicopter
6 - Fighter (Jet)

DATA SOURCE:
The data used is the MAD (Military Audio Dataset) dataset created by June-Woo Kim, Chihyeon Yoon and Ho-Young Jung
Github link to the open source data used: https://github.com/kaen2891/military_audio_dataset.git
Original article: https://www.nature.com/articles/s41597-024-03511-w

METHODOLOGY:
From the paper above, the researcher tested multiple different types of neural networks for machine learning, those being CNN-N, ResNet, EfficientNet, and AST. CNN was chosen to create my own NN due to its low number of parameters used and it acheiving some of the highest accuracies, pretrained and from scratch, of the many variations of the NN's tested by the researchers.
The main ML model created is "classify_data.py". Each audio file is taken and the log-mel spectrogram is taken of it and padded with a fade-in/fadeout to smoothly make each sample 7 seconds long while avoid sharp drops in energy across all frequencies. Due to the amount of time available, I decided to create a CNN model with three 3x3 conv comboed with BN and ReLU and a 2x2 max-pool. A common sequence of layers that perform (Convolutional Layer -> Batch Normalization (adjust and scale activations before ReLU) -> ReLU Activation -> Max-Pooling to reduce spatial dimensions).

Then the 3D tensor is flattened into a 1D array of size 55,296. This makes up the first layer of the training model, it passes each sample through the 55,296 neurons into a second layer of 256 neurons which is then passed into a third layer of 128 neurons which is finally passed into a layer of 7 neurons representing the 7 classifications of audio types. Altogether there are approximately 14.48 million parameters. Adam optimizer was used with a learning rate of 0.001. A patience tolerance of 15 iterations was put in place so the training could end early if no new peak accuracy in running the model on test data occurred. the best model would be saved in "best_model.pth" and a graph of accuracy and loss for training and test data would be outputted as well.

RESULTS:
After many experiments I found that within the 30-40 epochs it took to reach an end, the CNN model consistently converged to around 98 -99% training accuracy. The maximum test accuracy consistently hovered between 88-90%. I assume that after the testing accuracy peaks, overfitting begins to occur and that explains why the test accuracy would begin to dip throughout the following epochs even as training accuracy continues to rise.

IMPROVEMENTS?
Currently I've explored changing the sample duration and smoothing out the padding of audio samples. In the future I would like to improve on the Adam optimizer.

REPRODUCIBILITY:
classify_data.py and any other scripts in this workspace should work by typing "python classify_data.py" in terminal. Just make sure that the packages pandas, numpy, librosa, scikit-learn, and matplotlib have been installed. Also, the code has hardcoded portions for pathfinding so those must be adjusted on each new machine.  
NREIP Fall 2025 Project

Make sure to change the paths in the files after copying the code.

TODO:
- work on the gnu python block in the GNU_work folder (PRIORITY)
- if i have time find a way to 'randomize' the training and testing datasets?


GNU Radio system:
continous real-time signal-> get 7 second 16kHz sample -> run through model -> final classification
Run the sample parsing in 1 second increments for overlap in sounds, any chains of the same classification means that class of audio occurred for that duration

BIG ISSUES:
- latency budget
- multi window inference
- single label or multi on each sample?
- thresholds per class for identification?
- train dist log mel may differ from irl audio
- shorter sounds may be diluted in a 7second clip
- buffering, back-pressure. Need a buffer ring, preallocate tensors?
- resampling to ensure 16kHz?
- switch from librosa to torchaudio for GNU radio?
- use tensorRT?
- add hysteresis params