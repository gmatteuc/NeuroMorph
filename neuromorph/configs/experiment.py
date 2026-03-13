from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class ModelConfig:
    capacity: int = 16
    latent_dims: int = 32
    context_dim: int = 10
    context_mode: str = "linear"


@dataclass(frozen=True)
class NoiseConfig:
    training_noise_level: float = 0.5
    kernel_size: int = 11
    sigma: float = 2.0
    correlated: bool = True
    uncorrelated: bool = True


@dataclass(frozen=True)
class ContextConfig:
    context_id: int = 2
    signal_level: float = 1.0
    noise_level: float = 0.1
    eval_signal_level: float = 10.0
    target_digit: int = 3


@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 1e-3
    batch_size: int = 128
    epochs_baseline: int = 50
    epochs_association: int = 100
    seed: int = 42


@dataclass(frozen=True)
class DiscriminatorConfig:
    digit_a: int = 3
    digit_b: int = 4
    epochs: int = 200
    lr: float = 1e-3


@dataclass(frozen=True)
class PsychometricConfig:
    num_levels: int = 51
    num_samples: int = 5
    latent_noise_level: float = 0.1
    n_bootstrap: int = 1000


@dataclass(frozen=True)
class BimodalContextConfig:
    digit_a: int = 3
    digit_b: int = 4
    context_id: int = 2
    signal_level: float = 1.0
    noise_level: float = 0.1
    epochs_association: int = 100


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    psychometric: PsychometricConfig = field(default_factory=PsychometricConfig)
    bimodal: BimodalContextConfig = field(default_factory=BimodalContextConfig)
    pretrained_dir: str = "./pretrained"
    output_dir: str = "./output"

    def ensure_dirs(self):
        os.makedirs(self.pretrained_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
