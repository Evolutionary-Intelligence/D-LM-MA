# D-LM-MA (Distributed Low-Memory Matrix Adaptation evolution strategy)

The goal of this open-source repository is to provide all the source code and data involved in the paper ```Collective Learning of Low-Memory Matrix Adaptation for Large-Scale Black-Box Optimization```, which has been submitted to [PPSN-2022](https://ppsn2022.cs.tu-dortmund.de/) (*Under Review*).

## How-to-Run On Modern Cluster Computing

### Configurations of Python Programming Environment

It is suggested to use [conda](https://docs.conda.io/projects/conda/en/latest/index.html) to create the virtual environment for [Python](https://www.python.org/).

Here, we only need to install two very popular libraries for numerical computing and distributed computing: [NumPy](https://numpy.org/) and [Ray](https://www.ray.io/).

```bash
$ conda create --prefix env_ppsn -y  # for virtual environment
$ conda activate env_ppsn/
$ conda install --prefix env_ppsn python=3.8.12 -y  # for Python
$ pip install numpy==1.21.5  # for numerical computing
$ pip install ray==1.9.1  # for distributed computing
$ pip install "ray[default]"
```

### Settings of Environment Variables

According to the [official suggestions](https://docs.ray.io/en/latest/ray-core/troubleshooting.html#no-speedup) from ray, we set the following environment variables, in order to avoid contention with multi-threaded libraries (NumPy here).

```bash
$ export OPENBLAS_NUM_THREADS=1
$ export MKL_NUM_THREADS=1
$ export OMP_NUM_THREADS=1
$ export NUMEXPR_NUM_THREADS=1
```

Note that the above settings should be applied in all computing nodes.
