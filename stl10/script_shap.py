import os

import matplotlib.pyplot as plt
import numpy as np
import shap

from tqdm import tqdm

import instances

from models import load_model


# PARAMETERS
INDICES = range(0, 7999, 160)  # data samples
NUM_SAMPLES = 1024


if __name__ == "__main__":
    # LOAD MODEL
    model = load_model()

    # LOAD INSTANCES
    generator = instances.load_generator()
    traindata = instances.load_train_generator()
    for INDEX in tqdm(INDICES, "data sample"):
        x = generator[INDEX][0, ...]
        xname = os.path.splitext(os.path.split(generator.filenames[INDEX])[1])[
            0
        ]
        savedir = os.path.join("results", xname)
        os.makedirs(savedir, exist_ok=True)
        pred = model.predict(np.expand_dims(x, 0))
        node = np.argmax(pred[0, ...])

        # RUN SHAP METHOD
        imgs = np.concatenate(
            [np.mean(traindata[k], axis=0, keepdims=True) for k in range(8)],
            axis=0,
        )
        deep_exp = shap.DeepExplainer(model, imgs)
        mapping = np.reshape(
            deep_exp.explainer.shap_values(
                np.expand_dims(x, 0), check_additivity=False,
            )[node],
            x.shape,
        )

        plt.imsave(
            os.path.join(savedir, "shap.png",),
            np.mean(mapping, axis=-1),
            cmap="RdBu_r" if mapping.min() < 0 else "Reds",
            vmin=-np.abs(mapping).max() if mapping.min() < 0 else 0.0,
            vmax=np.abs(mapping).max(),
            format="png",
        )
        np.savez_compressed(
            os.path.join(savedir, "shap.npz",),
            **{
                "mapping": mapping,
                "method": "shap",
                "index": INDEX,
                "node": node,
                "prediction": pred,
            }
        )
