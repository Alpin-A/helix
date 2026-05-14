"""Train ACT policy on the pusht dataset."""
import json
from pathlib import Path

import torch
from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features


def make_delta_timestamps(delta_indices, fps):
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def main():
    output_directory = Path("outputs/act_pusht")
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps")
    dataset_id = "lerobot/pusht"

    print(f"Loading dataset: {dataset_id}")
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    print(f"Input features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")

    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    policy = ACTPolicy(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)

    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    training_steps = 5000
    log_freq = 100
    training_log = []

    print(f"Starting training for {training_steps} steps...")
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                loss_val = loss.item()
                training_log.append({"step": step, "loss": loss_val})
                print(f"step: {step} loss: {loss_val:.3f}")

            step += 1
            if step >= training_steps:
                done = True
                break

    log_path = output_directory / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to {log_path}")

    print("Training complete. Saving policy...")
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print(f"Saved to {output_directory}")


if __name__ == "__main__":
    main()