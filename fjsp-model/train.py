from fjsp_env import *
from fjsp_model.modules import *
from fjsp_model.utils import *
from lightning import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar



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
    1e-8,
    8,
    5,
    270,
    240,
    24,
    (128, 96, 96),
    (768, 512, 512),
    1024,
    5,
    256,
    (512, 328, 328),
    (5, 4, 4),
    768,
    8,
    768,
    6,
    1024,
    6,
    16,
    64,
    5,
    Agent.TrainStage.encode,
)

# model.load(
#     r"lightning_logs\version_9\checkpoints\epoch=14-step=150.ckpt",
#     Agent.TrainStage.policy,
# )



model.compile_modules()
torch.set_float32_matmul_precision("medium")

trainer = Trainer(
    # accelerator="cpu",
    callbacks=[TQDMProgressBar(leave=True)],
    log_every_n_steps=1,
    check_val_every_n_epoch=15,
    num_sanity_val_steps=0,
    max_epochs=-1,
)
trainer.fit(
    model,
    # ckpt_path=r"lightning_logs\version_10\checkpoints\epoch=29-step=300.ckpt",
)
