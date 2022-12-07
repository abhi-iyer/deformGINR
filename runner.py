from imports import *

from utils import *
from models import *
from data import *

from ginr_experiment import *
from ginr_configs import CONFIGS

bunny_3_fourier = GINR_Experiment("bunny_3_fourier", CONFIGS["bunny_3_fourier"])
bunny_3_fourier.train(num_epochs=5000, show_plot=False)

bunny_5_fourier = GINR_Experiment("bunny_5_fourier", CONFIGS["bunny_5_fourier"])
bunny_5_fourier.train(num_epochs=5000, show_plot=False)

bunny_10_fourier = GINR_Experiment("bunny_10_fourier", CONFIGS["bunny_10_fourier"])
bunny_10_fourier.train(num_epochs=5000, show_plot=False)

bunny_50_fourier = GINR_Experiment("bunny_50_fourier", CONFIGS["bunny_50_fourier"])
bunny_50_fourier.train(num_epochs=5000, show_plot=False)

bunny_100_fourier = GINR_Experiment("bunny_100_fourier", CONFIGS["bunny_100_fourier"])
bunny_100_fourier.train(num_epochs=5000, show_plot=False)