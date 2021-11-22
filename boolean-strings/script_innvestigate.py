import os
import sys

import innvestigate
import numpy as np

from tqdm import tqdm

import instances

from models import create_scaled_kmodel, create_simple_kmodel


# PARAMETERS
DIMENSION = 16
BLOCK_SIZE = 5
SAVEDIR = os.path.join("results")


if __name__ == "__main__":
    # LOAD MODEL
    if len(sys.argv) == 1:
        eta = 1.0
        model = create_simple_kmodel(
            dimension=DIMENSION, block_size=BLOCK_SIZE
        )
    else:
        eta = float(sys.argv[1])
        model = create_scaled_kmodel(
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

    # RUN INNVESTIGATE METHODS
    methods = [
        # ('lrp.epsilon', {'epsilon': 0.2}, 'LRP-E'),
        # ('lrp.sequential_preset_a', {'epsilon': 0.2}, 'LRP-A'),
        # ('lrp.sequential_preset_b', {'epsilon': 0.2}, 'LRP-B'),
        # ('lrp.sequential_preset_a_flat', {'epsilon': 0.2}, 'LRP-A-flat'),
        # ('lrp.sequential_preset_b_flat', {'epsilon': 0.2}, 'LRP-B-flat'),
        # ('deep_taylor.bounded', {'low': 0, 'high': 1}, 'DeepTaylor'),
        # ('deconvnet', {}, 'Deconvolution'),
        # ('gradient', {'postprocess': 'square'}, 'Sensitivity'),
        (
            "smoothgrad",
            {"noise_scale": 0.5, "augment_by_n": 64, "postprocess": "square"},
            "SmoothGrad",
        ),
        # ('input_t_gradient', {}, 'Input-x-Gradient'),
        # ('guided_backprop', {}, 'Guided-Backprop'),
        # ('integrated_gradients', {'reference_inputs': 0, 'steps': 64},
        #  'Integrated-Gradients'),
        # ('random', {'stddev': 0.5}, 'Input-plus-Random'),
        # ('input', {}, 'Input'),
    ]

    for method in tqdm(methods, "mapping methods"):
        inn_analyzer = innvestigate.create_analyzer(
            method[0], model, **method[1]
        )
        mapping = inn_analyzer.analyze(
            np.reshape(x, [1, DIMENSION, 1])
        ).squeeze()

        # STORE RESULTS
        os.makedirs(SAVEDIR, exist_ok=True)
        savepath = os.path.join(
            SAVEDIR, "bernoulli_eta_{:.2f}_{}.npz".format(eta, method[2],),
        )
        # savepath = os.path.join(
        #     SAVEDIR,
        #     'designed_eta_{:.2f}_{}.npz'.format(
        #         eta,
        #         method[2],
        #     ),
        # )
        np.savez_compressed(
            savepath,
            **{
                "mapping": mapping,
                "signal": x,
                "method": method[2],
                "eta": eta,
                "settings": method[1],
            }
        )
