[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://github.com/zhululai/auto-model-gen-sr/blob/master/LICENSE-CODE)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://github.com/zhululai/auto-model-gen-sr/blob/master/LICENSE-DATA)
![Python: 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)

# Automated Model Generation through State Reconstruction

This repository is created to share Python scripts, Arena instances and event logs for reproducing experiment results reported in our paper:

* L. Zhu, G. Lugaresi and A. Matta. **Automated Generation of Digital Models for Production Lines through State Reconstruction**. Accepted by *the 19th IEEE International Conference on Automation Science and Engineering*.

## Guide

The scripts have been tested with Python 3.9 and require the availability of `pandas`, `networkx`, `pygraphviz`, `numpy` and `matplotlib`. Apart from these Python libraries, you also need to install the latest version of [Graphviz](https://graphviz.org/download) (make sure that *Add Graphviz to the system PATH* is selected in the case of Windows).

Once the runtime environment has been set up, you can try the examples by executing

    python run_examples.py
The following command allows you to draw the results based on models that we generated:

    python run_experiments.py
If you want to redo the experiments, uncomment from line 18 to 28 and then execute this command, which may take up to a few hours depending on the performance of the host.

## License

All the code and data are released under the [BSD 3-Clause](https://github.com/zhululai/auto-model-gen-sr/blob/master/LICENSE-CODE) and [CC BY 4.0](https://github.com/zhululai/auto-model-gen-sr/blob/master/LICENSE-DATA) licenses respectively.
