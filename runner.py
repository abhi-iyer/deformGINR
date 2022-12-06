from imports import *

from utils import *
from models import *
from data import *

from ginr_experiment import *
from ginr_configs import CONFIGS

regular_bunny_3_fourier = GINR_Experiment("regular_bunny_3_fourier", CONFIGS["regular_bunny_3_fourier"])
regular_bunny_3_fourier.train(num_epochs=5000, show_plot=False)

regular_bunny_10_fourier = GINR_Experiment("regular_bunny_10_fourier", CONFIGS["regular_bunny_10_fourier"])
regular_bunny_10_fourier.train(num_epochs=5000, show_plot=False)

# regular_bunny_20_fourier = GINR_Experiment("regular_bunny_20_fourier", CONFIGS["regular_bunny_20_fourier"])
# regular_bunny_20_fourier.train(num_epochs=5000, show_plot=False)

# regular_bunny_30_fourier = GINR_Experiment("regular_bunny_30_fourier", CONFIGS["regular_bunny_30_fourier"])
# regular_bunny_30_fourier.train(num_epochs=5000, show_plot=False)

# regular_bunny_40_fourier = GINR_Experiment("regular_bunny_40_fourier", CONFIGS["regular_bunny_40_fourier"])
# regular_bunny_40_fourier.train(num_epochs=5000, show_plot=False)

# regular_bunny_50_fourier = GINR_Experiment("regular_bunny_50_fourier", CONFIGS["regular_bunny_50_fourier"])
# regular_bunny_50_fourier.train(num_epochs=5000, show_plot=False)

# regular_bunny_60_fourier = GINR_Experiment("regular_bunny_60_fourier", CONFIGS["regular_bunny_60_fourier"])
# regular_bunny_60_fourier.train(num_epochs=5000, show_plot=False)

# regular_bunny_70_fourier = GINR_Experiment("regular_bunny_70_fourier", CONFIGS["regular_bunny_70_fourier"])
# regular_bunny_70_fourier.train(num_epochs=5000, show_plot=False)

# regular_bunny_80_fourier = GINR_Experiment("regular_bunny_80_fourier", CONFIGS["regular_bunny_80_fourier"])
# regular_bunny_80_fourier.train(num_epochs=5000, show_plot=False)

# regular_bunny_90_fourier = GINR_Experiment("regular_bunny_90_fourier", CONFIGS["regular_bunny_90_fourier"])
# regular_bunny_90_fourier.train(num_epochs=5000, show_plot=False)

# regular_bunny_100_fourier = GINR_Experiment("regular_bunny_100_fourier", CONFIGS["regular_bunny_100_fourier"])
# regular_bunny_100_fourier.train(num_epochs=10000, show_plot=False)