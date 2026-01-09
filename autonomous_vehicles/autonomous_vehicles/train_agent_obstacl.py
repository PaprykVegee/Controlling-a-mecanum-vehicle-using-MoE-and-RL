import torch
from torch import nn
import segmentation_models_pytorch as smp

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

import gym
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
import os
from car_env_obstacle import GazeboLidarMaskEnvObstacle

# =========================
# Ustawienia
# =========================
ENV_PARALLEL = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOTAL_STEPS = 1_000_000
MAX_STEPS_PER_EPISODE = 1800
TIME_STEP = 0.1

config = {
    "algo": "PPO",
    "total_timesteps": TOTAL_STEPS,
    "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
    "n_steps": 512,
    "batch_size": 32,
    "n_epochs": 6,
    "lr": 1e-4,
    "gamma": 0.995,
    "env_parallel": ENV_PARALLEL,
}

# =========================
# Wandb init
# =========================
wandb.login()
run = wandb.init(
    project="MOE",
    entity="deep-neural-network-course",
    name='MOE1', 
    settings=wandb.Settings(save_code=False),
    config=config,
    sync_tensorboard=True,
    monitor_gym=False,
    save_code=False,
    mode='online'
)

# =========================
# Ładowanie UNet
# =========================
unet = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=8
).to(DEVICE)

checkpoint = torch.load(
    "/home/developer/ros2_ws/src/UNET_trening/best-unet-epoch=05-val_dice=0.9838.ckpt",
    map_location=DEVICE
)

state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("model."):
        new_state_dict[k[len("model."):]] = v
    else:
        new_state_dict[k] = v

unet.load_state_dict(new_state_dict, strict=False)
unet.eval()

# =========================
# Feature extractor dla agenta
# =========================
class AgentFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, model):
        super().__init__(observation_space, features_dim=1)
        device = next(model.parameters()).device  

        self.encoder = model.encoder

        old_conv1 = self.encoder.conv1
        new_conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=(old_conv1.bias is not None),
        )

        with torch.no_grad():
            if old_conv1.weight.shape[1] >= 2:
                new_conv1.weight[:, :2].copy_(old_conv1.weight[:, :2])
            else:
                new_conv1.weight[:, 0].copy_(old_conv1.weight[:, 0])
                new_conv1.weight[:, 1].copy_(old_conv1.weight[:, 0])
            if new_conv1.bias is not None and old_conv1.bias is not None:
                new_conv1.bias.copy_(old_conv1.bias)

        self.encoder.conv1 = new_conv1.to(device)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        with torch.no_grad():
            sample = torch.zeros((1,) + observation_space.shape, device=device)
            feats = self.encoder(sample)
            if isinstance(feats, list):
                feats = feats[-1]
            feats = self.pool(feats)
            self._features_dim = feats.view(1, -1).shape[1]

    def forward(self, x):
        #x = x / 255.0
        feats = self.encoder(x)
        if isinstance(feats, list):
            feats = feats[-1]
        feats = self.pool(feats)
        return feats.flatten(start_dim=1)

# =========================
# Parametry sieci głowicy
# =========================
head_arch = dict(
    pi=[512, 128, 64],
    vf=[256, 128, 64]
)

policy_kwargs = dict(
    features_extractor_class=AgentFeatureExtractor,
    features_extractor_kwargs=dict(
        model=unet
    ),
    net_arch=head_arch
)

# =========================
# Środowisko
# =========================
env_fn = lambda: GazeboLidarMaskEnvObstacle(max_steps=MAX_STEPS_PER_EPISODE, time_step=TIME_STEP)
vec_env = make_vec_env(env_fn, n_envs=ENV_PARALLEL)
vec_env = VecMonitor(vec_env)

tb_dir = os.path.join("tb_runs", run.id)

# =========================
# Inicjalizacja PPO
# =========================
model = PPO(
    policy="CnnPolicy",
    env=vec_env,
    n_steps=512,
    batch_size=32,
    n_epochs=6,
    learning_rate=1e-4,
    gamma=0.995,
    policy_kwargs=policy_kwargs,
    device=DEVICE,
    verbose=2,
    tensorboard_log=tb_dir,
)

model = PPO.load("/home/developer/ros2_ws/src/cheakpoints/checkpoint_295000_steps.zip", env=vec_env, device=DEVICE)

# =========================
# Callback do zapisywania checkpointów
# =========================
class SaveCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            checkpoint_file = os.path.join(self.save_path, f"checkpoint_{self.num_timesteps}_steps")
            self.model.save(checkpoint_file)
            if self.verbose > 0:
                print(f"Saved checkpoint: {checkpoint_file}")
        return True

# Tworzymy callbacki
checkpoint_callback = SaveCheckpointCallback(
    save_freq=5000, 
    save_path="/home/developer/ros2_ws/src/cheakpoints",
    verbose=2
)

wandb_callback = WandbCallback(
    model_save_path=os.path.join("wandb_models", run.id),
    model_save_freq=50_000,
    verbose=2,  
)

callback = CallbackList([wandb_callback, checkpoint_callback])

# =========================
# Trening
# =========================
model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

# =========================
# Zapis końcowy
# =========================
model.save("ppo_gazebo_unet_encoder")
run.finish()



################################## Ewaluacja ############################################

# from stable_baselines3 import PPO

# model = PPO.load(
#     "/home/developer/ros2_ws/src/cheakpoints/checkpoint_185000_steps.zip",
#     env=vec_env,
#     device=DEVICE
# )

# # 🔒 WYŁĄCZ TRYB TRENINGOWY
# model.policy.set_training_mode(False)

# # (opcjonalnie, ale dobre)
# vec_env.training = False
# vec_env.norm_reward = False

# obs = vec_env.reset()

# while True:
#     # 🔥 TU JEST CAŁA MAGIA
#     action, _ = model.predict(
#         obs,
#         deterministic=True
#     )

#     obs, reward, done, info = vec_env.step(action)

#     if done:
#         obs = vec_env.reset()

