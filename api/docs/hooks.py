import importlib
import sys
sys.path.append('..')
import remeta.configuration


def on_pre_build(config):
    # importlib.invalidate_caches()
    importlib.reload(remeta.configuration)
    # import remeta.configuration