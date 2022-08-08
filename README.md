# Generative Monte Carlo Particle Collisions

Monte Carlo and Generative model simulation of Electron Positron particle collision for the course Machine Learning in Particle Physics & Astronomy at Radboud University.

## How To Reproduce

1. Install the required packages: pandas, numpy, matplotlib, torch, scipy, sklearn

2. Run the MCS: `python3 lepton.py`

3. Run the training notebook `train.ipynb`, e.g. using `jupyter nbconvert --to notebook --inplace --execute GM.ipynb`. Set `train_X = True` where `X` in {VAE, GAN, NF} in the first cell in order to retrain the models, this takes approximately 10 minutes. When set to `False`, the saved models used in the paper are loaded.

4. Run the evaluation notebook `eval.ipynb` in a jupyter app or from the command line, similarly to the previous step.
