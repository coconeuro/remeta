Go directly to:
- [**Start page**](https://github.com/m-guggenmos/remeta/)
- [**Installation** (this page)](https://github.com/m-guggenmos/remeta/blob/master/INSTALL.md)
- [**Basic Usage**](https://github.com/m-guggenmos/remeta/blob/master/demo/basic_usage.ipynb)
- [**Common use cases**](https://github.com/m-guggenmos/remeta/blob/master/demo/common_use_cases.ipynb)
- [**Exotic use cases**](https://github.com/m-guggenmos/remeta/blob/master/demo/exotic_use_cases.ipynb)

# ReMeta Toolbox: installation

Remeta requires a working Python installation. It should run with Python >=3.9.

The ReMeta itself can be installed with `pip`:
```
pip install remeta
```

Or directly from GitHub:
```
pip install git+https://github.com/m-guggenmos/remeta.git
```
(this command requires an installed Git, e.g. [gitforwindows](https://gitforwindows.org/))


Required packages (should be automatically installed with pip):
- numpy (>=1.18.1)
- scipy (>=1.3)
- multiprocessing_on_dill (>=3.5.0a4) (only necessary for when the toolbox should be used with multiple cores)
- matplotlib (>=3.1.3)