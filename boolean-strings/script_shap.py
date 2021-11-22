import os
import sys

import numpy as np
import shap

import instances

from models import create_scaled_model, create_simple_model


# PARAMETERS
DIMENSION = 16
BLOCK_SIZE = 5
NUM_SAMPLES = 1024
SAVEDIR = os.path.join("results")


if __name__ == "__main__":
    # LOAD MODEL
    if len(sys.argv) == 1:
        eta = 1.0
        model = create_simple_model(dimension=DIMENSION, block_size=BLOCK_SIZE)
    else:
        eta = float(sys.argv[1])
        model = create_scaled_model(
            width=eta, dimension=DIMENSION, block_size=BLOCK_SIZE
        )

    # LOAD INSTANCES
    np.random.seed(1)
    x, pos = instances.create_bernoulli_bg_block(
        dimension=DIMENSION, block_size=BLOCK_SIZE, p=0.05,
    )
    # x, pos = instances.create_almost_block_bg_block(
    #     dimension=DIMENSION,
    #     block_size=BLOCK_SIZE,
    # )

    # RUN SHAP METHOD
    imgs = np.random.rand(NUM_SAMPLES, DIMENSION, 1)
    # imgs = np.random.binomial(1, 0.5 , (NUM_SAMPLES, DIMENSION, 1))
    deep_exp = shap.DeepExplainer(model, imgs)
    mapping = deep_exp.shap_values(np.reshape(x, [1, DIMENSION, 1]))[
        0
    ].squeeze()

    # STORE RESULTS
    os.makedirs(SAVEDIR, exist_ok=True)
    savepath = os.path.join(
        SAVEDIR, "bernoulli_eta_{:.2f}_shap.npz".format(eta),
    )
    # savepath = os.path.join(
    #     SAVEDIR,
    #     'designed_eta_{:.2f}_shap.npz'.format(
    #         eta,
    #     ),
    # )
    np.savez_compressed(
        savepath,
        **{"mapping": mapping, "signal": x, "method": "shap", "eta": eta}
    )
