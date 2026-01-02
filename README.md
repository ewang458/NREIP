# NREIP
*Fall 2025 NREIP Internship Project under the guidance of Dr. Aaron Cohen.

INSTALLATION:
1. git clone https://github.com/ewang458/NREIP.git
2. Create a python environment and install pandas, torch, numpy, librosa, scikit-learn, and matplotlib

GET STARTED:

Running ML Model:
1. Activate Python environment
2. change paths in classify_data.py to your pathways
3. python3 classify_data.py

Running the GNU Radio System:
1. Activate Python environment
2. gnuradio-companion

TODO:
- Create filter for noise
- Train model with shorter audio segments
- make multiple datasets to check for overfitting?

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

RESULTS:
After many experiments I found that within the 30-40 epochs it took to reach an end, the CNN model consistently converged to around 98 -99% training accuracy. The maximum test accuracy consistently hovered between 88-90%. I assume that after the testing accuracy peaks, overfitting begins to occur and that explains why the test accuracy would begin to dip throughout the following epochs even as training accuracy continues to rise.

REPRODUCIBILITY:
classify_data.py and any other scripts in this workspace should work by typing "python classify_data.py" in terminal. Just make sure that the packages pandas, numpy, librosa, scikit-learn, and matplotlib have been installed. Also, the code has hardcoded portions for pathfinding so those must be adjusted on each new machine.  
NREIP Fall 2025 Project

Make sure to change the paths in the files after copying the code.
