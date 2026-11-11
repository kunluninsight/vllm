try:
    from hisim._version import __version__
except ImportError:  # source tree that has never been built
    __version__ = "0.0.0+unknown"
