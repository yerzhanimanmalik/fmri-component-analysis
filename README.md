# fmri-component-analysis

Table of contents:

1. DL methods for Blind Source Separation:

  https://github.com/sigsep/open-unmix-pytorch
  One of the winners of MUSDB18 signal separation competition
  
  https://youtu.be/Xr7UOWIniCM
  
Open-Unmix is based on a three-layer bidirectional deep LSTM. The model learns to predict the magnitude spectrogram of a target, like vocals, from the magnitude spectrogram of a mixture input. Internally, the prediction is obtained by applying a mask on the input. The model is optimized in the magnitude domain using mean squared error and the actual separation is done in a post-processing step involving a multichannel wiener filter implemented using norbert. To perform separation into multiple sources, multiple models are trained for each particular target. While this makes the training less comfortable, it allows great flexibility to customize the training data for each target source.
  
  
2. ANICA. Adversarial Non-linear ICA:

http://github.com/anica Maximizing Independence with GANs for Non-linear ICA. Implemented in 2017, for Python2 and Tensorflow1



3. ICE-BeeM: Identifiable Conditional Energy-Based Deep Models Based on Nonlinear ICA. https://github.com/ilkhem/icebeem

Under certain constraints, non-linearly mixed sources are proven to be uniquely identifiable:
 - Nonstationarity (time-contrastive learning)   https://arxiv.org/abs/1605.06336
 - Temporal dependencies (permutation-contrastive learning)     https://arxiv.org/pdf/1805.08651.pdf
 - Existence of auxiliary variable (e.g. iVAE)    https://arxiv.org/abs/1907.04809


4. Vanilla VAE for 2d slices of task-based fMRI.





