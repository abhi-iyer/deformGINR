from imports import *

from utils import *
from models import *
from data import *

from ginr_experiment import *
from ginr_configs import CONFIGS


basic_bunny = GINR_Experiment("basic_bunny", CONFIGS["bunny_config"])
basic_bunny.train(num_epochs=5000, show_plot=False)