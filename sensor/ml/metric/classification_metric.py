from sensor.entity.artifact_entity import ClassificationMetricArtifact
from sensor.exception import SensorException
from sensor.logger import logging
import os
import sys
from sklearn.metrics import precision_score, f1_score, recall_score


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        classification_metric = ClassificationMetricArtifact(
            f1_score=f1, precision_score=precision, recall_score=recall)
        return classification_metric
    except Exception as e:
        raise SensorException(e, sys)
