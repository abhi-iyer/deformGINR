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
        n_nodes_in_sample=5000,
    ),
    model_class=GINR,
    optimizer_class=optim.AdamW,
    optimizer_args=dict(
        lr=1e-4,
    ),
    loss_fn=nn.CrossEntropyLoss(),
)

regular_bunny_10_fourier = deepcopy(regular_bunny_n_fourier)
regular_bunny_10_fourier["dataset_args"].update(n_fourier=10)

regular_bunny_20_fourier = deepcopy(regular_bunny_n_fourier)
regular_bunny_20_fourier["dataset_args"].update(n_fourier=20)

regular_bunny_30_fourier = deepcopy(regular_bunny_n_fourier)
regular_bunny_30_fourier["dataset_args"].update(n_fourier=30)

regular_bunny_40_fourier = deepcopy(regular_bunny_n_fourier)
regular_bunny_40_fourier["dataset_args"].update(n_fourier=40)

regular_bunny_50_fourier = deepcopy(regular_bunny_n_fourier)
regular_bunny_50_fourier["dataset_args"].update(n_fourier=50)

regular_bunny_60_fourier = deepcopy(regular_bunny_n_fourier)
regular_bunny_60_fourier["dataset_args"].update(n_fourier=60)

regular_bunny_70_fourier = deepcopy(regular_bunny_n_fourier)
regular_bunny_70_fourier["dataset_args"].update(n_fourier=70)

regular_bunny_80_fourier = deepcopy(regular_bunny_n_fourier)
regular_bunny_80_fourier["dataset_args"].update(n_fourier=80)

regular_bunny_90_fourier = deepcopy(regular_bunny_n_fourier)
regular_bunny_90_fourier["dataset_args"].update(n_fourier=90)

regular_bunny_100_fourier = deepcopy(regular_bunny_n_fourier)
regular_bunny_100_fourier["dataset_args"].update(n_fourier=100)


CONFIGS = dict(
    regular_bunny_10_fourier=regular_bunny_10_fourier,
    regular_bunny_20_fourier=regular_bunny_20_fourier,
    regular_bunny_30_fourier=regular_bunny_30_fourier,
    regular_bunny_40_fourier=regular_bunny_40_fourier,
    regular_bunny_50_fourier=regular_bunny_50_fourier,
    regular_bunny_60_fourier=regular_bunny_60_fourier,
    regular_bunny_70_fourier=regular_bunny_70_fourier,
    regular_bunny_80_fourier=regular_bunny_80_fourier,
    regular_bunny_90_fourier=regular_bunny_90_fourier,
    regular_bunny_100_fourier=regular_bunny_100_fourier,
)
