import os

import matplotlib.pyplot as plt
import numpy as np

from lime.lime_image import LimeImageExplainer
from tqdm import tqdm

import instances

from models import load_model


# PARAMETERS
INDICES = range(0, 1200, 100)  # data samples


if __name__ == "__main__":
    # LOAD MODEL
    model = load_model()

    # LOAD INSTANCES
    generator, _ = instances.load_generator()
    for INDEX in tqdm(INDICES, "data sample"):
        x = generator[INDEX][0, ...]
        xname = os.path.splitext(os.path.split(generator.filenames[INDEX])[1])[
            0
        ]
        savedir = os.path.join("results", xname)
        os.makedirs(savedir, exist_ok=True)
        pred = model.predict(np.expand_dims(x, 0))
        node = np.argmax(pred[0, ...])

        # RUN LIME METHOD
        explainer = LimeImageExplainer()
        explanations = explainer.explain_instance(x, model.predict,)
        explanation = explanations.local_exp[node]
        seg = explanations.segments
        mapping = np.zeros(x.shape)
        for pos, val in explanation:
            mapping[seg == pos] = val
        mapping = np.reshape(mapping, x.shape)

        plt.imsave(
            os.path.join(savedir, "lime.png",),
            np.mean(mapping, axis=-1),
            cmap="RdBu_r" if mapping.min() < 0 else "Reds",
            vmin=-np.abs(mapping).max() if mapping.min() < 0 else 0.0,
            vmax=np.abs(mapping).max(),
            format="png",
        )
        np.savez_compressed(
            os.path.join(savedir, "lime.npz",),
            **{
                "mapping": mapping,
                "method": "lime",
                "index": INDEX,
                "node": node,
                "prediction": pred,
            }
        )
