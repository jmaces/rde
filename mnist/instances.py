import importlib
import os

import numpy as np
import statstream
import statstream.approximate
import statstream.exact


DATAPATH = os.path.join("data", "keras_generators.py")
STATISTICSPATH = os.path.join(os.path.split(DATAPATH)[0], "statistics")


def load_generator(path=DATAPATH, class_mode=None):
    generator_path = os.path.expanduser(path)
    generator_spec = importlib.util.spec_from_file_location("", generator_path)
    generator_module = importlib.util.module_from_spec(generator_spec)
    generator_spec.loader.exec_module(generator_module)
    return generator_module.load_test_data(class_mode=class_mode)


def load_train_generator(path=DATAPATH, class_mode=None):
    generator_path = os.path.expanduser(path)
    generator_spec = importlib.util.spec_from_file_location("", generator_path)
    generator_module = importlib.util.module_from_spec(generator_spec)
    generator_spec.loader.exec_module(generator_module)
    return generator_module.load_train_data_augmented(class_mode=class_mode)


def load_statistics(path=STATISTICSPATH, mode="diag", rank=30):
    print("Dataset statistics will be loaded now.")
    try:
        if mode == "half":
            loaded = np.load(
                os.path.join(path, "{}-mode-rank-{}.npz".format(mode, rank))
            )
        else:
            loaded = np.load(os.path.join(path, "{}-mode.npz".format(mode)))
        return loaded["mean"], loaded["covariance"]
    except Exception:
        if mode == "half":
            print(
                "No precomputed dataset statistics for {} mode with rank {} "
                "were found at {}.".format(mode, path)
            )
        else:
            print(
                "No precomputed dataset statistics for {} mode were "
                "found at {}.".format(mode, path)
            )

        reply = (
            str(input("Do you want to compute it now (y/n)?")).lower().strip()
        )
        if reply in ["y", "yes"]:
            compute_statistics(path, mode, rank)
            load_statistics(path, mode, rank)


def compute_statistics(path=STATISTICSPATH, mode="diag", rank=30):
    print(
        "Statistics will be calculated now. Depending on the dataset "
        "size and covariance calculation mode this may take a while."
    )
    generator = load_train_generator()
    os.makedirs(path, exist_ok=True)
    if mode == "diag":
        savepath = os.path.join(path, "{}-mode.npz".format(mode),)
        mean, cov = statstream.streaming_mean_and_variance(
            generator, steps=generator.n // generator.batch_size,
        )
    elif mode == "half":
        savepath = os.path.join(
            path, "{}-mode-rank-{}.npz".format(mode, rank),
        )
        mean, cov = statstream.streaming_mean_and_low_rank_covariance(
            generator,
            rank,
            steps=generator.n // generator.batch_size,
            tree=True,
            reset=None,
        )
    elif mode == "full":
        savepath = os.path.join(path, "{}-mode.npz".format(mode),)
        mean, cov = statstream.streaming_mean_and_covariance(
            generator, steps=generator.n // generator.batch_size,
        )
    else:
        raise ValueError(
            "Unknown covariance calculation " "mode {}.".format(mode)
        )
    np.savez_compressed(
        savepath, **{"mean": mean, "covariance": cov, "mode": mode}
    )
    print(
        "Finished computing statistics. Results saved at {}.".format(savepath)
    )
