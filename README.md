# counterix
[![GitHub release][release-image]][release-url]
[![PyPI release][pypi-image]][pypi-url]
[![Build][build-image]][build-url]
[![MIT License][license-image]][license-url]


[release-image]:https://img.shields.io/github/release/akb89/counterix.svg?style=flat-square
[release-url]:https://github.com/akb89/counterix/releases/latest
[pypi-image]:https://img.shields.io/pypi/v/counterix.svg?style=flat-square
[pypi-url]:https://pypi.org/project/counterix/
[build-image]:https://img.shields.io/github/workflow/status/akb89/counterix/CI?style=flat-square
[build-url]:https://github.com/akb89/counterix/actions?query=workflow%3ACI
[license-image]:http://img.shields.io/badge/license-MIT-000000.svg?style=flat-square
[license-url]:LICENSE.txt

A small toolkit to generate count-based PPMI-weighed SVD Distributional Semantic Models.

## Install
```shell
pip install counterix
```

or, after a git clone:
```shell
python3 setup.py install
```

## Use

### Generate
To generate a raw count matrix from a tokenized corpus, run:
```
counterix generate \
  --corpus /abs/path/to/corpus/txt/file \
  --min-count frequency_threshold \
  --win-size window_size \
  --output abs/path/to/output/model/directory
```

If the `--output` parameter is not set, the output files will be saved to the corpus directory.

### Weigh
To weigh a raw count model with PPMI, run:
```
counterix weigh \
  --model /abs/path/to/raw/count/npz/model \
  --output /abs/path/to/output/model/directory \
  --weighing-func ppmi
```

### SVD
To apply SVD on a PPMI-weighed model, with k=10000, run:
```
counterix svd \
  --model /abs/path/to/ppmi/npz/model \
  --dim 10000 \
  --which LM  # largest singular values
```

## Warning
For informativeness, np.dot in `prob_values = np.exp(np.dot(l1, self._model.trainables.syn1neg.T))` using multithreading by default with openblas. Need to run the code with `env OMP_NUM_THREADS=1 counterix generate...`
