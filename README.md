[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)
![Python: 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)

# Automatic Model Generation through State Reconstruction

This repository is created to share Python scripts, Arena instances and event logs for reproducing experiment results reported in our paper:

* L. Zhu, G. Lugaresi and A. Matta. **Automatic Generation of Digital Twins for Manufacturing Systems through State Reconstruction**. Submitted to *the 19th IEEE International Conference on Automation Science and Engineering*.

## Guide

The scripts have been tested with Python 3.9 on Windows and require the installation of `pandas`, `networkx`, `pygraphwiz`, `numpy` and `matplotlib`. Apart from these Python libraries, you also need to download the latest [Graphwiz Archive](https://graphviz.org/download/#windows) and uncompress it in the same folder. 

Once the runtime environment has been set up, you can try the examples by executing

    python run_examples.py
The following command allows you to draw the results based on models that we generated:

    python run_experiments.py
If you want to redo the experiments, uncomment from line 18 to 28 and then execute this command, which may take up to a few hours depending on the performance of the host.

## License

All the code and data are released under the [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
