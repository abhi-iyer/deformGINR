from imports import *
from models import *
from data import *
from utils import *

base = dict()

bunny_config = deepcopy(base)
bunny_config.update(
    batch_size=32,
    dataset_class=GraphDataset,
    dataset_args=dict(
        dataset_dir="./data",
    ),
    model_class=GINR,
    optimizer_class=optim.AdamW,
    optimizer_args=dict(
        lr=1e-4,
    ),
    scheduler_class=lr_scheduler.ReduceLROnPlateau,
    scheduler_args=dict(
        factor=0.5,
        patience=5000,
    ),
    loss_fn=nn.CrossEntropyLoss(),
)


CONFIGS = dict(
    bunny_config=bunny_config,
)