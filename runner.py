from imports import *

from utils import *
from models import *
from data import *

from ginr_experiment import *
from ginr_configs import CONFIGS

regular_bunny_3_fourier = GINR_Experiment("regular_bunny_3_fourier", CONFIGS["regular_bunny_3_fourier"])
regular_bunny_3_fourier.train(num_epochs=5000, show_plot=False)

regular_bunny_5_fourier = GINR_Experiment("regular_bunny_5_fourier", CONFIGS["regular_bunny_5_fourier"])
regular_bunny_5_fourier.train(num_epochs=5000, show_plot=False)

regular_bunny_10_fourier = GINR_Experiment("regular_bunny_10_fourier", CONFIGS["regular_bunny_10_fourier"])
regular_bunny_10_fourier.train(num_epochs=5000, show_plot=False)

regular_bunny_100_fourier = GINR_Experiment("regular_bunny_100_fourier", CONFIGS["regular_bunny_100_fourier"])
regular_bunny_100_fourier.train(num_epochs=5000, show_plot=False)