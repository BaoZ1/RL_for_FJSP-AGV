from fjsp_env import *
from modules import *
from utils import *
from lightning import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint


train_env = Environment(
    32,
    [
        GenerateParam(
            5,
            4,
            2,
            3,
            1,
            4,
            0.7,
            5,
            7,
            False,
        ),
        GenerateParam(
            5,
            4,
            2,
            3,
            1,
            4,
            0.7,
            5,
            7,
            True,
        ),
        GenerateParam(
            10,
            8,
            7,
            3,
            4,
            10,
            0.7,
            2,
            8,
            False,
        ),
        GenerateParam(
            10,
            8,
            7,
            3,
            4,
            10,
            0.7,
            2,
            8,
            False,
        ),
        GenerateParam(
            17,
            8,
            5,
            4,
            1,
            5,
            0.7,
            2,
            7,
            False,
        ),
        GenerateParam(
            17,
            8,
            5,
            4,
            1,
            5,
            0.7,
            2,
            7,
            True,
        ),
    ],
    True,
)


model = Agent(
    train_env,
    1e-5,
    8,
    5,
    270,
    240,
    24,
    (64, 48, 48),
    (256, 192, 192),
    512,
    5,
    128,
    (256, 192, 192),
    (5, 4, 4),
    384,
    4,
    512,
    5,
    512,
    6,
    4,
    32,
    5,
    Agent.TrainStage.explore,
)

model.load(
    r"lightning_logs\version_3\checkpoints\policy-on-v2-epoch=47-step=480.ckpt",
    Agent.TrainStage.policy,
)


model.compile_modules()
torch.set_float32_matmul_precision("medium")

trainer = Trainer(
    
    # accelerator="cpu",
    callbacks=[TQDMProgressBar(leave=True), ModelCheckpoint(filename='explore-on-v3-{epoch}-{step}')],
    log_every_n_steps=1,
    check_val_every_n_epoch=12,
    num_sanity_val_steps=0,
    max_epochs=-1,
)
trainer.fit(
    model,
    # ckpt_path=r"lightning_logs\version_0\checkpoints\encode.ckpt",
)
