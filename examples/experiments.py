import neptune
import torch
from avalanche.benchmarks import SplitMNIST
from avalanche.training import AGEM, EWC, GEM, LwF, Naive, Replay, SynapticIntelligence, CWRStar
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

from models.mlp import MLP

# Neptune config
PROJECT_NAME = "mateusz.wojcik.contact/continual-learning"

# Model architecture config
INPUT_LAYER_SIZE = 784
HIDDEN_LAYER_SIZE = 256
OUTPUT_LAYER_SIZE = 10

# Training config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LEARNING_RATE = 5e-5
TRAIN_EPOCHS = 1
DROP_RATE = 0.0
N_EXPERIENCES = 10
FIXED_CLASS_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

model_architecture_config = {
    "INPUT_LAYER_SIZE": INPUT_LAYER_SIZE,
    "HIDDEN_LAYER_SIZE": HIDDEN_LAYER_SIZE,
    "OUTPUT_LAYER_SIZE": OUTPUT_LAYER_SIZE
}

training_config = {
    "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
    "TEST_BATCH_SIZE": TEST_BATCH_SIZE,
    "LEARNING_RATE": LEARNING_RATE,
    "TRAIN_EPOCHS": TRAIN_EPOCHS,
    "DROP_RATE": DROP_RATE,
    "N_EXPERIENCES": N_EXPERIENCES,
    "DEVICE": DEVICE,
}

methods = {
    'Naive': (Naive, dict(), ["baseline"]),
    'LwF': (LwF, dict(alpha=0.15, temperature=1.5), ["regularization"]),
    'EWC': (EWC, dict(ewc_lambda=10_000), ["regularization"]),
    'SynapticIntelligence': (SynapticIntelligence, dict(si_lambda=5e7), ["regularization"]),
    'CWRStar': (CWRStar, dict(cwr_layer_name=None), ["architectural"]),
    'Replay_5': (Replay, dict(mem_size=5 * N_EXPERIENCES), ["memory"]),
    'GEM_5': (GEM, dict(patterns_per_exp=5, memory_strength=0.5), ["regularization", "memory"]),
    'AGEM_10': (AGEM, dict(patterns_per_exp=5, sample_size=20), ["regularization", "memory"]),
}


if __name__ == '__main__':
    print(f"Running on device {DEVICE}")

    for method_name, (method, method_arguments, tags) in tqdm(methods.items(), total=len(methods)):
        print(f"Start training {method_name} with setup: {method_arguments}")

        # Initialize Neptune run
        run = neptune.init_run(project=PROJECT_NAME, tags=tags)

        # Log the setup
        run["parameters"] = {
            'method': method.__name__,
            **method_arguments,
            **model_architecture_config,
            **training_config,
        }

        # Model.
        # To make experiments fair, remember to always initialize a new model and not to use already trained one.
        model = MLP(
            input_layer_size=INPUT_LAYER_SIZE,
            hidden_layer_size=HIDDEN_LAYER_SIZE,
            output_layer_size=OUTPUT_LAYER_SIZE,
            drop_rate=DROP_RATE,
        )

        # Prepare for training & testing
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = CrossEntropyLoss()

        # Training strategy
        strategy = method(
            model,
            optimizer,
            criterion,
            train_epochs=TRAIN_EPOCHS,
            train_mb_size=TRAIN_BATCH_SIZE,
            eval_mb_size=TEST_BATCH_SIZE,
            device=DEVICE,
            **method_arguments,
        )

        # Dataset
        mnist = SplitMNIST(n_experiences=N_EXPERIENCES, fixed_class_order=FIXED_CLASS_ORDER)
        train_stream = mnist.train_stream
        test_stream = mnist.test_stream

        # Train and test
        for train_task in train_stream:
            strategy.train(experiences=train_task)
            results = strategy.eval(test_stream)

            # Save main metric as test_score
            run[f"metrics/test_score"] = results['Top1_Acc_Stream/eval_phase/test_stream/Task000']

            # Save all metrics
            for metric, value in results.items():
                run[f"metrics/{metric}"].append(value)

        run.stop()
