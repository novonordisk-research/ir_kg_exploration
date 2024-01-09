from pykeen.hpo import hpo_pipeline

import os
import pickle
import numpy as np
import torch
import  wordcloud

import pykeen
from pykeen.datasets import OpenBioLink
from pykeen.pipeline import pipeline
from pykeen import predict
from pykeen.evaluation import RankBasedEvaluator  

from pykeen.datasets import get_dataset
from pykeen.triples import TriplesFactory

from pykeen.evaluation import RankBasedEvaluator




train = np.loadtxt('OpenBioLink_training_large.csv', dtype=str,delimiter=',')
val  = np.loadtxt('OpenBioLink_valdiation_large.csv', dtype=str,delimiter=',')
test  = np.loadtxt('OpenBioLink_test_large.csv', dtype=str,delimiter=',')

train = TriplesFactory.from_labeled_triples(train)
val = TriplesFactory.from_labeled_triples(val)
test = TriplesFactory.from_labeled_triples(test)



# Trains model with best hyperparamter can change to any model, hyperparamters are selected from best model

for i in range(10):
    result = pipeline(
        training=train,
        validation=val,
        testing=test,
        model='RotatE',
        model_kwargs=dict(embedding_dim=200),
        optimizer='Adam',
        optimizer_kwargs=dict(lr=0.01),
        loss='marginranking',
        training_loop='slcwa',
        training_kwargs=dict(num_epochs=300, batch_size=8192),
        negative_sampler='basic',
        negative_sampler_kwargs=dict(num_negs_per_pos=1),

        lr_scheduler='ExponentialLR',
        lr_scheduler_kwargs=dict(
            gamma=0.97
        ),
        evaluator_kwargs=dict(filtered=True),
        stopper='early',
        stopper_kwargs=dict(frequency=5,patience=2,relative_delta=0.002)


    )

    # How to look at the model
    model = result.model
    model

    with open("models/RotateE_result_large"+str(i)+".pkl", "wb") as f:
        pickle.dump(model, f)
