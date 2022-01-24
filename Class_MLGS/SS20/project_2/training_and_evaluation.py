from typing import Callable, Union, Tuple, List, Dict
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
from models import SmoothClassifier
from torch.utils.data import RandomSampler

def train_model(model: nn.Module, dataset: Dataset, batch_size: int, loss_function: Callable, optimizer: Optimizer,
                epochs: int = 1, loss_args: Union[dict, None] = None) -> Tuple[List, List]:
    """
    Train a model on the input dataset.
    Parameters
    ----------
    model: nn.Module
        The input model to be trained.
        Default: ConvNN()
    dataset: torch.utils.data.Dataset
        The dataset to train on.
        Default: mnist_trainset, len=60000
    batch_size: int
        The training batch size.
        Default: 128
    loss_function: function with signature: (x, y, model, **kwargs) -> (loss, logits).
        The function used to compute the loss.device = model.device()
        Default: see the notation as above
    optimizer: Optimizer
        The model's optimizer.
        Default: opt = Adam(model.parameters(), lr=lr)
    epochs: int
        Number of epochs to train for.
        Default: 1.
    loss_args: dict or None
        Additional arguments to be passed to the loss function.

    Returns
    -------
    Tuple containing
        * losses: List[float]. The losses obtained at each step.
        * accuracies: List[float]. The accuracies obtained at each step.

    """
    if loss_args is None:
        loss_args = {}
    losses = []
    accuracies = []
    num_train_batches = int(torch.ceil(torch.tensor(len(dataset) / batch_size)).item()) # 60000/128=469
    for epoch in range(epochs):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # len=469
        for x, y in tqdm(iter(train_loader), total=num_train_batches):
            ##########################################################
            # x: torch.Size([128, 1, 28, 28])
            # y: torch.Size([128])
            # YOUR CODE HERE

            device = model.device()
            x, y = x.to(device), y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # if len(loss_args):
            #     loss, logits = loss_function(x, y, model, epsilon=loss_args["epsilon"], norm=loss_args["norm"])
            # else:
            #     loss, logits = loss_function(x, y, model)
            loss, logits = loss_function(x, y, model, **loss_args)
            top1 = torch.argmax(logits, dim=1)
            n_correct = torch.sum(top1 == y)
            accuracy = n_correct.item() / len(x)

            # in train process, need to optimize the parameters
            loss.backward()
            optimizer.step()

            losses.append(loss)
            accuracies.append(accuracy)
            ##########################################################
    return losses, accuracies


def predict_model(model: nn.Module, dataset: Dataset, batch_size: int, attack_function: Union[Callable, None] = None,
                  attack_args: Union[Callable, None] = None) -> float:
    """
    Use the model to predict a label for each sample in the provided dataset. Optionally performs an attack via
    the attack function first.
    Parameters
    ----------
    model: nn.Module
        The input model to be used.
        Default: ConvNN
    dataset: torch.utils.data.Dataset
        The dataset to predict for.
        Default: mnist_testet, len=10000
    batch_size: int
        The batch size.
        Default: test_batch_size, 1000, feel free to change
    attack_function: function or None
        If not None, call the function to obtain a perturbed batch before evaluating the prediction.
    attack_args: dict or None
        Additionall arguments to be passed to the attack function.

    Returns
    -------
    float: the accuracy on the provided dataset.
    """
    if attack_args is None:
        attack_args = {}
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_batches = int(torch.ceil(torch.tensor(len(dataset) / batch_size)).item())
    predictions = []
    targets = []
    for x, y in tqdm(iter(test_loader), total=num_batches):
        ##########################################################
        # YOUR CODE HERE

        x.requires_grad = True
        device = model.device()
        x, y = x.to(device), y.to(device)

        logits = model.forward(x)

        if attack_function is not None:
            model.zero_grad()
            x_pert = attack_function(logits, x, y, attack_args["epsilon"], attack_args["norm"])
            logits = model(x_pert)

        pred_result = torch.argmax(logits, dim=1)
        target = y

        predictions.append(pred_result)
        targets.append(target)
        ##########################################################
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    accuracy = (predictions == targets).float().mean().item()
    return accuracy


def evaluate_robustness_smoothing(base_classifier: nn.Module, sigma: float, dataset: Dataset,
                                  num_samples_1: int = 50, num_samples_2: int = 500,
                                  alpha: float = 0.05, certification_batch_size: float = 5000, num_classes: int = 10
                                  ) -> Dict:
    # remember to change the num_samples_2 back when doing the final evaluation
    """
    Evaluate the robustness of a smooth classifier based on the input base classifier via randomized smoothing.
    Parameters
    ----------
    base_classifier: nn.Module
        The input base classifier to use in the randomized smoothing process.
    sigma: float
        The variance to use for the Gaussian noise samples.
    dataset: Dataset
        The input dataset to predict on.
    num_samples_1: int
        The number of samples used to determine the most likely class.
    num_samples_2: int
        The number of samples used to perform the certification.
    alpha: float
        The desired confidence level that the top class is indeed the most likely class. E.g. alpha=0.05 means that
        the expected error rate must not be larger than 5%.
    certification_batch_size: int
        The batch size to use during the certification, i.e. how many noise samples to classify in parallel.
    num_classes: int
        The number of classes.

    Returns
    -------
    Dict containing the following keys:
        * abstains: int. The number of times the smooth classifier abstained, i.e. could not certify the input sample to
                    the desired confidence level.
        * false_predictions: int. The number of times the prediction could be certified but was not correct.
        * correct_certified: int. The number of times the prediction could be certified and was correct.
        * avg_radius: float. The average radius for which the predictions could be certified.

    """
    Sampler = RandomSampler(dataset, replacement=True, num_samples=1000)
    model = SmoothClassifier(base_classifier=base_classifier, sigma=sigma, num_classes=num_classes)
    # test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=Sampler)
    abstains = 0
    false_predictions = 0
    correct_certified = 0
    radii = []
    # for x, y in tqdm(iter(test_loader), total=len(dataset)):
    for x, y in tqdm(iter(test_loader), total=1000):
        ##########################################################
        # YOUR CODE HERE
        top_class, radius = model.certify(x, num_samples_1, num_samples_2, alpha, certification_batch_size)
        # radii.append(radius)
        if top_class == -1:
            abstains += 1
        else:
            radii.append(radius)
            winning_class = model.predict(x, num_samples_1, alpha, certification_batch_size)
            print("real class:", y, ", top_class(model.certify):", top_class, ", and winning_class(model.predict):", winning_class)
            if top_class == y:
                correct_certified += 1
            else:
                false_predictions += 1
        ##########################################################
    avg_radius = torch.tensor(radii).mean().item()
    return dict(abstains=abstains, false_predictions=false_predictions, correct_certified=correct_certified,
                avg_radius=avg_radius)