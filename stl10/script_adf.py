import os

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

import instances

from models import load_adfmodel, load_model
from rde import adf_rde_lagrangian


# PARAMETERS
MODE = "half"  # 'diag' or 'half'
RANK = 30  # only affects 'half' mode
INDICES = range(0, 7999, 160)  # data samples
MUS = [1e-2, 1e-1, 1e0]  # affects Lagrangian optimization

if __name__ == "__main__":
    # LOAD MODEL
    adfmodel = load_adfmodel(mode=MODE, rank=RANK)
    model = load_model()

    # LOAD STATISTICS
    mean, covariance = instances.load_statistics(mode=MODE, rank=RANK)

    # LOAD INSTANCES
    generator = instances.load_generator()
    for INDEX in tqdm(INDICES, "data sample"):
        x = generator[INDEX][0, ...]
        xname = os.path.splitext(os.path.split(generator.filenames[INDEX])[1])[
            0
        ]
        savedir = os.path.join("results", xname)
        os.makedirs(savedir, exist_ok=True)
        pred = model.predict(np.expand_dims(x, 0))
        node_second, node_first = np.argpartition(pred[0, ...], -2)[-2:]
        target = pred[0, node_first]
        second = pred[0, node_second]

        # RUN ADF-RDE METHOD
        for MU in MUS:
            s0 = 0.25 * np.ones((np.prod(x.shape),))
            mapping = np.reshape(
                adf_rde_lagrangian(
                    np.expand_dims(x, 0),
                    node_first,
                    target,
                    s0.flatten(),
                    np.expand_dims(mean, 0),
                    np.expand_dims(covariance, 0),
                    adfmodel,
                    MU,
                    mode=MODE,
                ),
                x.shape,
            )
            plt.imsave(
                os.path.join(
                    savedir,
                    "{}-mode-lagrangian-lagrangian_{}.png".format(MODE, MU),
                ),
                np.mean(mapping, axis=-1),
                cmap="Reds",
                vmin=0.0,
                vmax=1.0,
                format="png",
            )
            np.savez_compressed(
                os.path.join(
                    savedir,
                    "{}-mode-lagrangian-lagrangian_{}.npz".format(MODE, MU),
                ),
                **{
                    "mapping": mapping,
                    "method": "{}-mode-lagrangian".format(MODE),
                    "regfactor": MU,
                    "index": INDEX,
                    "mode": MODE,
                    "node": node_first,
                    "prediction": pred,
                    "rank": RANK,
                }
            )
