from pathlib import Path
import pickle
import numpy as np

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


# Select dataset name (OpenBioLink, Hetionet) and dataset type (small, large)
DATASET_NAME = "OpenBioLink"
DATASET_TYPE = "large"

# Change model to model of choice, i.e., RotatE, TransE, CompGCN, etc.
MODEL_NAME = "RotatE"

train = np.loadtxt(
    f"{DATASET_NAME}_training_{DATASET_TYPE}.csv", dtype=str, delimiter=","
)
val = np.loadtxt(
    f"{DATASET_NAME}_valdiation_{DATASET_TYPE}.csv", dtype=str, delimiter=","
)
test = np.loadtxt(f"{DATASET_NAME}_test_{DATASET_TYPE}.csv", dtype=str, delimiter=",")

train = TriplesFactory.from_labeled_triples(train)
val = TriplesFactory.from_labeled_triples(val)
test = TriplesFactory.from_labeled_triples(test)


# Trains model with best hyperparamter can change to any model, hyperparamters are selected from best model
for i in range(10):
    result = pipeline(
        training=train,
        validation=val,
        testing=test,
        model=MODEL_NAME,
        model_kwargs=dict(embedding_dim=200),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=0.01),
        loss="marginranking",
        training_loop="slcwa",
        training_kwargs=dict(num_epochs=300, batch_size=8192),
        negative_sampler="basic",
        negative_sampler_kwargs=dict(num_negs_per_pos=1),
        lr_scheduler="ExponentialLR",
        lr_scheduler_kwargs=dict(gamma=0.97),
        evaluator_kwargs=dict(filtered=True),
        stopper="early",
        stopper_kwargs=dict(frequency=5, patience=2, relative_delta=0.002),
    )

    # How to look at the model
    model = result.model
    model

    Path(f"models/{DATASET_NAME}_{DATASET_TYPE}/{MODEL_NAME}").mkdir(
        parents=True, exist_ok=True
    )
    with open(
        f"models/{DATASET_NAME}_{DATASET_TYPE}/{MODEL_NAME}/{MODEL_NAME}_result"
        + str(i)
        + ".pkl",
        "wb",
    ) as f:
        pickle.dump(model, f)
