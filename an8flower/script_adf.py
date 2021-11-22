import os

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

import instances

from models import load_adfmodel, load_model
from rde import adf_rde_lagrangian


# PARAMETERS
MODE = "half"  # 'diag', 'half', or 'full'
RANK = 10  # only affects 'half' mode
INDICES = range(0, 1200, 100)  # data samples
MUS = [5e0]  # affects Lagrangian optimization

if __name__ == "__main__":
    # LOAD MODEL
    adfmodel = load_adfmodel(mode=MODE, rank=RANK)
    model = load_model()

    # LOAD STATISTICS
    mean, covariance = instances.load_statistics(mode=MODE, rank=RANK)

    # LOAD INSTANCES
    generator, mask_generator = instances.load_generator()
    for INDEX in tqdm(INDICES, "data sample"):
        x = generator[INDEX][0, ...]
        mask = 1 - mask_generator[INDEX][0, ...]
        xname = os.path.splitext(os.path.split(generator.filenames[INDEX])[1])[
            0
        ]
        savedir = os.path.join("results", xname)
        os.makedirs(savedir, exist_ok=True)
        pred = model.predict(np.expand_dims(x, 0))
        node_second, node_first = np.argpartition(pred[0, ...], -2)[-2:]
        target = pred[0, node_first]
        second = pred[0, node_second]

        # store reference mask for comparison
        plt.imsave(
            os.path.join(savedir, "mask.png",),
            np.mean(mask, axis=-1),
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            format="png",
        )
        np.savez_compressed(
            os.path.join(savedir, "mask.npz",),
            **{
                "mapping": mask,
                "method": "mask",
                "index": INDEX,
                "node": node_first,
                "prediction": pred,
            }
        )

        print("predicted class of x:\t {}".format(node_first))
        print("pre softmax score:\t {}".format(target))
        print("gap to second best:\t {}".format(target - second))

        # RUN ADF-RDE METHOD
        for MU in MUS:
            s0 = np.clip(
                0.5 * np.ones((np.prod(x.shape),))
                + 1e-1 * np.random.randn(np.prod(x.shape)),
                0.0,
                1.0,
            )
            mapping = np.reshape(
                adf_rde_lagrangian(
                    np.expand_dims(x, 0),
                    node_first,
                    target,
                    s0,
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
                    savedir, "{}-mode-lagrangian_{}.png".format(MODE, MU),
                ),
                np.mean(mapping, axis=-1),
                cmap="Reds",
                vmin=0.0,
                vmax=1.0,
                format="png",
            )
            np.savez_compressed(
                os.path.join(
                    savedir, "{}-mode-lagrangian_{}.npz".format(MODE, MU),
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
