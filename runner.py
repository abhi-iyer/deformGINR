from imports import *

from utils import *
from models import *
from data import *

from ginr_experiment import *
from ginr_configs import CONFIGS


deformed_3_fourier = GINR_Experiment("deformed_3_fourier", CONFIGS["deformed_3_fourier"])
deformed_3_fourier.train(num_epochs=1, show_plot=False)

deformed_5_fourier = GINR_Experiment("deformed_5_fourier", CONFIGS["deformed_5_fourier"])
deformed_5_fourier.train(num_epochs=1, show_plot=False)

deformed_10_fourier = GINR_Experiment("deformed_10_fourier", CONFIGS["deformed_10_fourier"])
deformed_10_fourier.train(num_epochs=1, show_plot=False)

deformed_50_fourier = GINR_Experiment("deformed_50_fourier", CONFIGS["deformed_50_fourier"])
deformed_50_fourier.train(num_epochs=1, show_plot=False)

deformed_100_fourier = GINR_Experiment("deformed_100_fourier", CONFIGS["deformed_100_fourier"])
deformed_100_fourier.train(num_epochs=1, show_plot=False)



perturbed_3_fourier = GINR_Experiment("perturbed_3_fourier", CONFIGS["perturbed_3_fourier"])
perturbed_3_fourier.train(num_epochs=1, show_plot=False)

perturbed_5_fourier = GINR_Experiment("perturbed_5_fourier", CONFIGS["perturbed_5_fourier"])
perturbed_5_fourier.train(num_epochs=1, show_plot=False)

perturbed_10_fourier = GINR_Experiment("perturbed_10_fourier", CONFIGS["perturbed_10_fourier"])
perturbed_10_fourier.train(num_epochs=1, show_plot=False)

perturbed_50_fourier = GINR_Experiment("perturbed_50_fourier", CONFIGS["perturbed_50_fourier"])
perturbed_50_fourier.train(num_epochs=1, show_plot=False)

perturbed_100_fourier = GINR_Experiment("perturbed_100_fourier", CONFIGS["perturbed_100_fourier"])
perturbed_100_fourier.train(num_epochs=1, show_plot=False)



unperturbed_3_fourier = GINR_Experiment("unperturbed_3_fourier", CONFIGS["unperturbed_3_fourier"])
unperturbed_3_fourier.train(num_epochs=1, show_plot=False)

unperturbed_5_fourier = GINR_Experiment("unperturbed_5_fourier", CONFIGS["unperturbed_5_fourier"])
unperturbed_5_fourier.train(num_epochs=1, show_plot=False)

unperturbed_10_fourier = GINR_Experiment("unperturbed_10_fourier", CONFIGS["unperturbed_10_fourier"])
unperturbed_10_fourier.train(num_epochs=1, show_plot=False)

unperturbed_50_fourier = GINR_Experiment("unperturbed_50_fourier", CONFIGS["unperturbed_50_fourier"])
unperturbed_50_fourier.train(num_epochs=1, show_plot=False)

unperturbed_100_fourier = GINR_Experiment("unperturbed_100_fourier", CONFIGS["unperturbed_100_fourier"])
unperturbed_100_fourier.train(num_epochs=1, show_plot=False)