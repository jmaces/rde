import os

import innvestigate
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

import instances

from models import load_kmodel


# PARAMETERS
INDICES = range(0, 1200, 100)  # data samples


if __name__ == "__main__":
    # LOAD MODEL
    model = load_kmodel()

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

        # RUN INNVESTIGATE METHODS
        methods = [
            # ('lrp.epsilon', {'epsilon': 0.2}, 'LRP-E'),
            # ('lrp.sequential_preset_a', {'epsilon': 0.2}, 'LRP-A'),
            # ('lrp.sequential_preset_b', {'epsilon': 0.2}, 'LRP-B'),
            ("lrp.sequential_preset_a_flat", {"epsilon": 0.2}, "LRP-A-flat"),
            # ('lrp.sequential_preset_b_flat', {'epsilon': 0.2}, 'LRP-B-flat'),
            (
                "deep_taylor.bounded",
                {"low": x.min(), "high": x.max()},
                "DeepTaylor",
            ),
            # ('deconvnet', {}, 'Deconvolution'),
            ("gradient", {"postprocess": "square"}, "Sensitivity"),
            (
                "smoothgrad",
                {
                    "noise_scale": 0.3 * (x.max() - x.min()),
                    "augment_by_n": 64,
                    "postprocess": "square",
                },
                "SmoothGrad",
            ),
            # ('input_t_gradient', {}, 'Input-x-Gradient'),
            ("guided_backprop", {}, "Guided-Backprop"),
            # ('integrated_gradients', {'reference_inputs': x.min(),
            #  'steps': 64}, 'Integrated-Gradients'),
            # ('random', {'stddev': 2*np.std(x)}, 'Input-plus-Random'),
            # ('input', {}, 'Input'),
        ]

        for method in tqdm(methods, "mapping methods"):
            inn_analyzer = innvestigate.create_analyzer(
                method[0], model, neuron_selection_mode="index", **method[1]
            )
            mapping = np.reshape(
                inn_analyzer.analyze(np.expand_dims(x, 0), node,), x.shape,
            )
            plt.imsave(
                os.path.join(savedir, "{}.png".format(method[2]),),
                np.mean(mapping, axis=-1),
                cmap="RdBu_r" if mapping.min() < 0 else "Reds",
                vmin=-np.abs(mapping).max() if mapping.min() < 0 else 0.0,
                vmax=np.abs(mapping).max(),
                format="png",
            )
            np.savez_compressed(
                os.path.join(savedir, "{}.npz".format(method[2]),),
                **{
                    "mapping": mapping,
                    "method": method[2],
                    "index": INDEX,
                    "node": node,
                    "prediction": pred,
                }
            )
