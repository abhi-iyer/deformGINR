from imports import *
from models import *
from data import *
from utils import *

base = dict()

regular_bunny_n_fourier = deepcopy(base)
regular_bunny_n_fourier.update(
    batch_size=32,
    dataset_class=GraphDataset,
    dataset_args=dict(
        dataset_dir="./regular_bunny_data",
        n_fourier=10,
        n_nodes_in_sample=128,
    ),
    model_class=GINR,
    optimizer_class=optim.AdamW,
    optimizer_args=dict(
        lr=1e-4,
    ),
    loss_fn=nn.CrossEntropyLoss,
)

regular_bunny_3_fourier = deepcopy(regular_bunny_n_fourier)
regular_bunny_3_fourier["dataset_args"].update(n_fourier=3)

regular_bunny_5_fourier = deepcopy(regular_bunny_n_fourier)
regular_bunny_5_fourier["dataset_args"].update(n_fourier=5)

regular_bunny_10_fourier = deepcopy(regular_bunny_n_fourier)
regular_bunny_10_fourier["dataset_args"].update(n_fourier=10)

regular_bunny_100_fourier = deepcopy(regular_bunny_n_fourier)
regular_bunny_100_fourier["dataset_args"].update(n_fourier=100)


CONFIGS = dict(
    regular_bunny_3_fourier=regular_bunny_3_fourier,
    regular_bunny_5_fourier=regular_bunny_5_fourier,
    regular_bunny_10_fourier=regular_bunny_10_fourier,
    regular_bunny_100_fourier=regular_bunny_100_fourier,
)
