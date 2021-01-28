import warnings
warnings.filterwarnings("error")

warnings.warn("Deprecated", DeprecationWarning)
warnings.warn("dropout option adds dropout after all but last "
                "recurrent layer, so non-zero dropout expects "
                "num_layers greater than 1, but got dropout={} and "
                "num_layers={}".format(0.5, 1))