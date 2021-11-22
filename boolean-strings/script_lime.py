import os
import sys

import numpy as np

from lime.lime_tabular import LimeTabularExplainer

import instances

from models import create_scaled_model, create_simple_model


# PARAMETERS
DIMENSION = 16
BLOCK_SIZE = 5
NUM_SAMPLES = 1024
SAVEDIR = os.path.join("results")


def wrapped_prediction(x):
    preds = model.predict(np.expand_dims(x, -1))
    probs = np.concatenate([preds, 1 - preds], axis=1)
    return probs


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

    # RUN LIME METHOD
    explainer = LimeTabularExplainer(
        # training_data=np.random.binomial(1, 0.5, (NUM_SAMPLES, DIMENSION)),
        training_data=np.random.rand(NUM_SAMPLES, DIMENSION),
        mode="classification",
    )
    explanations = explainer.explain_instance(
        x, wrapped_prediction, top_labels=2,
    )
    pro_explanation = explanations.as_map()[0]
    pro_pos, pro_val = zip(*pro_explanation)
    pro_mapping = np.zeros(DIMENSION)
    pro_mapping[list(pro_pos)] = list(pro_val)
    # con_explanation = explanations.as_map()[1]
    # con_pos, con_val = zip(*con_explanation)
    # con_mapping = np.zeros(DIMENSION)
    # con_mapping[list(con_pos)] = list(con_val)

    # STORE RESULTS
    os.makedirs(SAVEDIR, exist_ok=True)
    savepath = os.path.join(
        SAVEDIR, "bernoulli_eta_{:.2f}_lime.npz".format(eta),
    )
    # savepath = os.path.join(
    #     SAVEDIR,
    #     'designed_eta_{:.2f}_lime.npz'.format(
    #         eta,
    #     ),
    # )
    np.savez_compressed(
        savepath,
        **{"mapping": pro_mapping, "signal": x, "method": "lime", "eta": eta}
    )
