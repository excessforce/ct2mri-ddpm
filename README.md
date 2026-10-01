# ct2mri-ddpm-main

An expansion on Rachel Gordon's GANCM CT-to-MRI synthesis work.

# Model

Standard DDPM. A noise scheduler with gaussian diffusion for the forward process, a U-Net neural network for the back process. We use a Deterministic DDIM sampler for quicker previews.

Note: Nothing regarding Polaris has been implemented yet. This should be done before experimentation begins.

# Preprocessing

The data is (supposedly) stored at "/media/aisec-102/DATA3/Rachel/data/CV/".
Ultimately, due to my limited access to the server, a lot of the architecture surrounding paths may need to be changed around. This will depend on how the folds are stored since I used argparse to read command-line input (especially train.py), so some issues there are to be expected.

# Current Scope

Currently, the model is designed with 256px images. Tests were already incredibly long on the GANCM model, and DDPM consumes much more memory due to its backward process.
U-Net will allow 512px images, and nothing technically stops it. But, the model was not designed with it in mind. Modifications to the training and loading routines should be made beforehand.