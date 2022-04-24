# D-LM-MA (Distributed Low-Memory Matrix Adaptation evolution strategy)

The goal of this open-source repository is to provide all the source code and data involved in the paper ```Collective Learning of Low-Memory Matrix Adaptation for Large-Scale Black-Box Optimization```, which has been submitted to [PPSN-2022](https://ppsn2022.cs.tu-dortmund.de/) (*Under Review*).

## How-to-Run On Modern Cluster Computing

### Configurations of the Python Programming Environment

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

### Construction of a Private Cluster Computing Platform

When ```CentOS``` is used, the firewall should be closed in advance for all nodes, as presented below.

```bash
$ sudo firewall-cmd --state
$ sudo systemctl stop firewalld
$ sudo systemctl disable firewalld
$ sudo systemctl mask --now firewalld
$ sudo firewall-cmd --state
```

We need to run one private Ray Cluster where the distributed algorithm is executed. There are several ways to construct it. For more guidelines about the Ray Cluster, see [https://docs.ray.io/en/latest/cluster/quickstart.html](https://docs.ray.io/en/latest/cluster/quickstart.html). Here is the simplest way to construct it:

```bash
$ ray start --head  # run on the (single) master node, depending on your choice
# run on all slave nodes
$ ray start --address='[MASTER-ID:PORT]' --redis-password='[PASSWORD]'
# [MASTER-ID:PORT] and [PASSWORD] need to be replaced by your own settings
$ ray status  # run on the master node to check the status information of the Ray Cluster
```

### How to Run Trials

Since typically the optimization process in high-dimensional cases needs a **very long** runtime, it is better to run these algorithms *in the background* (e.g., via ```nohup```).

```bash
# run D-LM-MA (with 250 islands) independently for 7 times
$ nohup python run_trials.py -s=1 -e=7 -o=LMMAES -d=True -i=250 >DistributedLMMAES_1_7.out 2>&1 &
# run the baseline algorithm: serial MAES
$ nohup python run_trials.py -s=1 -e=7 -o=MAES >MAES_1_7.out 2>&1 &
# run the baseline algorithm: serial LMMAES
$ nohup python run_trials.py -s=1 -e=7 -o=LMMAES >LMMAES_1_7.out 2>&1 &
```

Note that the **core code** for D-LM-MA is available at [https://github.com/Evolutionary-Intelligence/D-LM-MA/blob/main/pypoplib/distributed_es.py](https://github.com/Evolutionary-Intelligence/D-LM-MA/blob/main/pypoplib/distributed_es.py), while ```distributed_lmmaes.py``` is just its wrapper based on LM-MA-ES. In fact, other LSO versions of CMA-ES can also be used here *with little modifications*.
