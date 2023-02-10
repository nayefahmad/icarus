from typing import List

from model_dispatch import models, pipelines


def train(pipeline_num: int, model_num: int) -> List:
    # take raw data, pass through pipeline

    # train:

    # test:

    # return preds and model:
    pass


num_pipelines = len(pipelines.keys())
num_models = len(models.keys())
num_combos = num_models * num_pipelines

combos = []
for pipe in pipelines.keys():
    for model in models.keys():
        combo_name = f"pipe-{pipe}_model-{model}"
        combos.append(combo_name)

results = {}
for pipe in pipelines.keys():
    for model in models.keys():
        preds, model = train(pipe, model)
        results[combos]
