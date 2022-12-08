from imports import *
from models import *
from data import *
from utils import *

base = dict()

bunny_n_fourier = deepcopy(base)
bunny_n_fourier.update(
    batch_size=256,
    dataset_class=GraphDataset,
    dataset_args=dict(
        dataset_dir="./bunny_data",
        n_fourier=10,
    ),
    model_class=GINR,
    optimizer_class=optim.AdamW,
    optimizer_args=dict(
        lr=1e-4,
    ),
    loss_fn=nn.CrossEntropyLoss,
)

bunny_3_fourier = deepcopy(bunny_n_fourier)
bunny_3_fourier["dataset_args"].update(n_fourier=3)

bunny_5_fourier = deepcopy(bunny_n_fourier)
bunny_5_fourier["dataset_args"].update(n_fourier=5)

bunny_10_fourier = deepcopy(bunny_n_fourier)
bunny_10_fourier["dataset_args"].update(n_fourier=10)

bunny_50_fourier = deepcopy(bunny_n_fourier)
bunny_50_fourier["dataset_args"].update(n_fourier=50)

bunny_100_fourier = deepcopy(bunny_n_fourier)
bunny_100_fourier["dataset_args"].update(n_fourier=100)


CONFIGS = dict(
    bunny_3_fourier=bunny_3_fourier,
    bunny_5_fourier=bunny_5_fourier,
    bunny_10_fourier=bunny_10_fourier,
    bunny_50_fourier=bunny_50_fourier,
    bunny_100_fourier=bunny_100_fourier,
)
