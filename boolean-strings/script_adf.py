import os
import sys

import numpy as np

import instances

from models import (
    create_scaled_adfmodel,
    create_scaled_model,
    create_simple_adfmodel,
    create_simple_model,
)
from rde import adf_rde_lagrangian


# PARAMETERS
DIMENSION = 16
BLOCK_SIZE = 5
MODE = "diag"  # 'diag', 'half', or 'full'
RANK = 16  # only affects 'half' mode
SAVEDIR = os.path.join("results")


if __name__ == "__main__":
    # LOAD MODEL
    if len(sys.argv) == 1:
        eta = 1.0
        adfmodel = create_simple_adfmodel(
            dimension=DIMENSION, block_size=BLOCK_SIZE, mode=MODE
        )
        model = create_simple_model(dimension=DIMENSION, block_size=BLOCK_SIZE)
    else:
        eta = float(sys.argv[1])
        adfmodel = create_scaled_adfmodel(
            width=eta, dimension=DIMENSION, block_size=BLOCK_SIZE, mode=MODE
        )
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
    pred = model.predict(np.reshape(x, [1, DIMENSION, 1]))

    # RUN ADF-RDE METHOD
    os.makedirs(SAVEDIR, exist_ok=True)
    mean = 0.5 * np.ones((1, DIMENSION, 1))
    if MODE == "diag":
        covariance = 1 / 12 * np.ones((1, DIMENSION, 1))
    elif MODE == "half":
        covariance = np.reshape(
            np.sqrt(1 / 12) * np.eye(RANK, DIMENSION), (1, RANK, DIMENSION, 1),
        )
    elif MODE == "full":
        covariance = np.reshape(
            1 / 12 * np.eye(DIMENSION, DIMENSION),
            (1, DIMENSION, 1, DIMENSION, 1),
        )
    s0 = mean.copy()

    for mu in np.logspace(-5, 0, 10):
        mapping = adf_rde_lagrangian(
            np.reshape(x, [1, DIMENSION, 1]),
            pred,
            s0,
            mean,
            covariance,
            adfmodel,
            mu,
            mode=MODE,
        )

        # STORE RESULTS
        savepath = os.path.join(
            SAVEDIR,
            "bernoulli_eta_{:.2f}_adf-rde-{}-lagrangian_{:1.2e}.npz".format(
                eta, MODE, mu,
            ),
        )
        # savepath = os.path.join(
        #     SAVEDIR,
        #     'designed_eta_{:.2f}_adf-rde-{}-lagrangian_{:1.2e}.npz'.format(
        #         eta,
        #         MODE,
        #         mu,
        #     ),
        # )
        np.savez_compressed(
            savepath,
            **{
                "mapping": mapping,
                "signal": x,
                "method": "adf-rde-{}-lagrangian".format(MODE),
                "mode": MODE,
                "eta": eta,
                "mu": mu,
            }
        )
