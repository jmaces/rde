# Rate-Distortion Explanations (RDE)

[![GitHub license](https://img.shields.io/github/license/jmaces/rde)](https://github.com/jmaces/rde/blob/master/LICENSE)
[![code-style black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-tensorflow](https://img.shields.io/badge/Made%20with-Tensorflow-1f425f.svg)](https://www.tensorflow.org/)
[![made-with-keras](https://img.shields.io/badge/Made%20with-Keras-1f425f.svg)](https://keras.io/)

This repository provides the official implementation of our papers and conference contributions regarding rate-distortion explanations:

- [Explaining Neural Network Decisions Is Hard](http://interpretable-ml.org/icml2020workshop/pdf/12.pdf)  
J. Macdonald, S. Wäldchen, S. Hauch and G. Kutyniok (2020)  
_ICML 2020 Workshop, XXAI: Extending Explainable AI Beyond Deep Models and Classifiers_
- [A Rate-Distortion Framework for Explaining Neural Network Decisions](https://arxiv.org/abs/1905.11092)  
J. Macdonald, S. Wäldchen, S. Hauch and G. Kutyniok (2019)

## Usage

Each sub-directory contains the code for a separate experiment.

#### What's Inside?
Individual experiments contain definitions of the used neural networks, scripts for loading the data, as well as scripts for computing and evaluating several interpretability methods.

#### What's Not Inside
The pretrained network weights are not included due to size restrictions.
The data sets for [mnist](http://yann.lecun.com/exdb/mnist/), [stl10](https://cs.stanford.edu/~acoates/stl10/), and [an8flower](https://euterpe.idlab.uantwerpen.be/~joramasmogrovejo/projects/visualExplanationByInterpretation/index.html) are not included in this repository but are publicly available at the respective websites.

#### Preparing The Data
The `boolean-string` experiment uses synthetically generated data and is completely self-contained within this repository.  
For the three other experiments, after downloading the data, it needs to be unpacked into the respective prepared `data` sub-directories in a format readable by `keras` data generators via `flow_from_directory` as specified in the `keras_generator.py` file (you can also move the data directory somewhere else and adapt the `DATAPATH` in the `instances.py` file).  
Data set statistics will have to be computed when you run a relevance mapping script for the first time (this is done using the [`statstream`](https://github.com/jmaces/statstream) package).


#### Computing Relevance Maps
Use the files `scripts_*.py` to compute relevance maps using different interpretability methods.

- `script_adf.py`: Our rate-distortion explanation (RDE) method using the assumed density filtering (ADF) implementation. Different variants are available (`diag`, `half` aka `low-rank`, and `full` which is only feasible for low-dimensional signals and small networks; we recommend to always use `diag` or `half`).
- `script_shap.py`: Shapley value based implementation by Scott Lundberg et al., see https://github.com/slundberg/shap.
- `script_lime.py`: Local Interpretable Model-Agnostic Explanations (LIME) implementation by Marco Ribeiro et al., see https://github.com/marcotcr/lime.
- `script_innvestigate.py`: Collection of sensitivity and backpropagation based interpretation method implementations by Maximilian Alber et al., see https://github.com/albermax/innvestigate.
    - Deep Taylor Decompositions
    - Guided Backprop
    - Layerwise relevance Propagation (LRP)
    - Sensitivity Analysis
    - SmoothGrad

#### Evaluating Relevance Maps
Use the files `pixelflip.py` and `pixelflip_visualize.py`.


## Requirements

The package versions are the ones we used. Other versions might work as well.

`cudatoolkit` *(v10.0.130)*  
`cudnn` *(v7.6.5)*  
`h5py` *(v2.10.0)*  
`hdf5` *(v1.10.5)*  
`innvestigate` *(v1.0.8)*  
`keras-adf` *(v19.1.0)*  
`keras` *(v.2.2.4)*  
`lime` *(v0.1.1.37)*  
`matplotlib` *(v3.1.2)*  
`numpy` *(v1.17)*  
`pillow` *(v8.0.1)*  
`python` *(v3.7.8)*  
`scikit-image` *(0.16.2)*  
`scipy` *(v1.4.1)*  
`shap` *(v0.34.0)*  
`statstream` *(19.1.0)*  
`tensorflow-gpu` *(v1.15)*  
`tikzplotlib` *(v0.9.8)*  
`tqdm` *(v4.53.0)*  


## License

This repository is MIT licensed, as found in the [LICENSE](LICENSE) file.
