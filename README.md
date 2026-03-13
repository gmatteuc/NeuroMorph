# NeuroMorph

This repository is a side project by Giulio Matteucci aimed at exploring multimodal associations using a convolutional denoising autoencoder. The model is trained on the MNIST dataset, with an added contextual input representing a secondary sensory modality. This context, consistently paired with a specific digit (e.g., "3"), acts as a prior to bias reconstruction under noisy conditions. The goal is to simulate how contextual cues influence perception, serving as a computational proof of concept for visuo-tactile association experiments in neuroscience. Training consists of a baseline phase for general denoising, followed by an association phase where the context strongly correlates with one digit to exert influence in ambiguous scenarios. This setup provides a foundational model for studying sensory integration and its potential effects on perception. In the future, more in-depth analysis of the contextual input’s effect on the autoencoder’s representations, as well as simulated psychophysics experiments, will be added.

- Contextual inputs reshape the visual representations of the autoencoder.
- Context alters the autoencoder’s reconstruction.
- Context biases the perceptual judgment of a visual classifier built on autoencoder representations.

![contextual_autoencoder](https://github.com/user-attachments/assets/91106d0d-da86-4f91-b7d8-db4a679a6292)

Here a presentation illustrating some of the key results: https://docs.google.com/presentation/d/1uNlbAPi-_Sjb5JjfNdis1I3rWV5Bi7w2/edit?usp=sharing&ouid=114959095852310266125&rtpof=true&sd=true

--------------------------------------------------------------------------------------------

TODO:
- Behavior:
i) Choose two categories (e.g., "3" and another digit).
ii) Use the vanilla autoencoder to generate a dataset of in-between stimuli via latent space interpolation (with added noise).
iii) Build a decision-maker by attaching a logistic regression unit to the latent space of the associative autoencoder.
iv) Construct psychometric curves based on the decision-maker’s performance at different levels of interpolation (and/or noise), with and without context.
v) Quantify the psychometric shift induced by the context.
- Physiology:
i) Conduct a sparse noise experiment in both modalities (alone and together) for units in the latent space.
ii) Quantify congruent/incongruent tuning and multisensory enhancement.
iii) Attempt to reconstruct the Most Exciting Input (MEI) via optimization.
- Bayesian Analysis:
Retrain another associative autoencoder with a bimodal prior (e.g., context is 50% correlated with "3" and 50% with "4"). Sample from the latent space with and without context, then create frequency histograms of all categories. Does the context result in a bimodal distribution, or does it collapse to an in-between, blurred average digit?



