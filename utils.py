import csv
import torch


def mean(x):
    return sum(x) / len(x)


def evaluate(true, pred):
    """
    evaluates accuracy of pathology classification
    :param true: ground truth labels of pathology
    :param pred: predicted one-hot approximation of classifier
    :return:
    """
    pred = torch.argmax(pred, dim=-1)
    acc = (true == pred).float().mean()
    return acc


def save_history(file, history, mode='w'):
    """
    writes history to a csv file
    :param file: name of the file
    :param history: list of history
    :param mode: writing mode
    :return: None
    """
    with open(file, mode) as f:
        writer = csv.writer(f)
        history = [line.replace(':', ',').split(',') for line in history]
        [writer.writerow(line) for line in history]
