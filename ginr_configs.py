from imports import *
from models import *
from data import *
from utils import *


base = dict()
base.update(
    batch_size=256,
    dataset_class=GraphDataset,
    dataset_args=dict(
        dataset_dir="",
        n_fourier=10,
    ),
    model_class=GINR,
    model_args=dict(
        hidden_dim=256,
        num_layers=4,
    ),
    optimizer_class=optim.AdamW,
    optimizer_args=dict(
        lr=1e-4,
    ),
    loss_fn=nn.CrossEntropyLoss,
)


deformed_5_fourier = deepcopy(base)
deformed_5_fourier["dataset_args"].update(dataset_dir="./deformed_bunny")
deformed_5_fourier["dataset_args"].update(n_fourier=5)

deformed_10_fourier = deepcopy(base)
deformed_10_fourier["dataset_args"].update(dataset_dir="./deformed_bunny")
deformed_10_fourier["dataset_args"].update(n_fourier=10)

deformed_50_fourier = deepcopy(base)
deformed_50_fourier["dataset_args"].update(dataset_dir="./deformed_bunny")
deformed_50_fourier["dataset_args"].update(n_fourier=50)

deformed_100_fourier = deepcopy(base)
deformed_100_fourier["dataset_args"].update(dataset_dir="./deformed_bunny")
deformed_100_fourier["dataset_args"].update(n_fourier=100)



perturbed_5_fourier = deepcopy(base)
perturbed_5_fourier["dataset_args"].update(dataset_dir="./perturbed_bunny")
perturbed_5_fourier["dataset_args"].update(n_fourier=5)

perturbed_10_fourier = deepcopy(base)
perturbed_10_fourier["dataset_args"].update(dataset_dir="./perturbed_bunny")
perturbed_10_fourier["dataset_args"].update(n_fourier=10)

perturbed_50_fourier = deepcopy(base)
perturbed_50_fourier["dataset_args"].update(dataset_dir="./perturbed_bunny")
perturbed_50_fourier["dataset_args"].update(n_fourier=50)

perturbed_100_fourier = deepcopy(base)
perturbed_100_fourier["dataset_args"].update(dataset_dir="./perturbed_bunny")
perturbed_100_fourier["dataset_args"].update(n_fourier=100)



unperturbed_5_fourier = deepcopy(base)
unperturbed_5_fourier["dataset_args"].update(dataset_dir="./unperturbed_bunny")
unperturbed_5_fourier["dataset_args"].update(n_fourier=5)

unperturbed_10_fourier = deepcopy(base)
unperturbed_10_fourier["dataset_args"].update(dataset_dir="./unperturbed_bunny")
unperturbed_10_fourier["dataset_args"].update(n_fourier=10)

unperturbed_50_fourier = deepcopy(base)
unperturbed_50_fourier["dataset_args"].update(dataset_dir="./unperturbed_bunny")
unperturbed_50_fourier["dataset_args"].update(n_fourier=50)

unperturbed_100_fourier = deepcopy(base)
unperturbed_100_fourier["dataset_args"].update(dataset_dir="./unperturbed_bunny")
unperturbed_100_fourier["dataset_args"].update(n_fourier=100)



CONFIGS = dict(
    deformed_5_fourier=deformed_5_fourier,
    deformed_10_fourier=deformed_10_fourier,
    deformed_50_fourier=deformed_50_fourier,
    deformed_100_fourier=deformed_100_fourier,

    perturbed_5_fourier=perturbed_5_fourier,
    perturbed_10_fourier=perturbed_10_fourier,
    perturbed_50_fourier=perturbed_50_fourier,
    perturbed_100_fourier=perturbed_100_fourier,

    unperturbed_5_fourier=unperturbed_5_fourier,
    unperturbed_10_fourier=unperturbed_10_fourier,
    unperturbed_50_fourier=unperturbed_50_fourier,
    unperturbed_100_fourier=unperturbed_100_fourier,
)
