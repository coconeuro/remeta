# Installation

Remeta requires a working Python installation. It should run with Python >=3.10.

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
- numpy (>=1.20)
- scipy (>=1.3)
- multiprocessing_on_dill (>=3.5.0a4) (only necessary for when the toolbox should be used with multiple cores)
- matplotlib (>=3.6.0)
- numdifftools (>=0.9.38)