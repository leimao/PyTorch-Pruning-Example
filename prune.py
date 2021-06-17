import os
import copy
import torch
import torch.nn.utils.prune as prune
from utils import set_random_seeds, create_model, prepare_dataloader, train_model, save_model, load_model, evaluate_model, create_classification_report
import argparse


def compute_final_pruning_rate(pruning_rate, num_iterations):
    """A function to compute the final pruning rate for iterative pruning.
        Note that this cannot be applied for global pruning rate if the pruning rate is heterogeneous among different layers.

    Args:
        pruning_rate (float): Pruning rate.
        num_iterations (int): Number of iterations.

    Returns:
        float: Final pruning rate.
    """

    final_pruning_rate = 1 - (1 - pruning_rate)**num_iterations

    return final_pruning_rate


def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def measure_global_sparsity(model,
                            weight=True,
                            bias=False,
                            conv2d_use_mask=False,
                            linear_use_mask=False):

    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

        elif isinstance(module, torch.nn.Linear):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def iterative_pruning_finetuning(model,
                                 train_loader,
                                 test_loader,
                                 device,
                                 learning_rate,
                                 l1_regularization_strength,
                                 l2_regularization_strength,
                                 learning_rate_decay=0.1,
                                 conv2d_prune_amount=0.4,
                                 linear_prune_amount=0.2,
                                 num_iterations=10,
                                 num_epochs_per_iteration=10,
                                 model_filename_prefix="pruned_model",
                                 model_dir="saved_models",
                                 grouped_pruning=False):

    for i in range(num_iterations):

        print("Pruning and Finetuning {}/{}".format(i + 1, num_iterations))

        print("Pruning...")

        if grouped_pruning == True:
            # Global pruning
            # I would rather call it grouped pruning.
            parameters_to_prune = []
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    parameters_to_prune.append((module, "weight"))
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=conv2d_prune_amount,
            )
        else:
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module,
                                          name="weight",
                                          amount=conv2d_prune_amount)
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module,
                                          name="weight",
                                          amount=linear_prune_amount)

        _, eval_accuracy = evaluate_model(model=model,
                                          test_loader=test_loader,
                                          device=device,
                                          criterion=None)

        classification_report = create_classification_report(
            model=model, test_loader=test_loader, device=device)

        num_zeros, num_elements, sparsity = measure_global_sparsity(
            model,
            weight=True,
            bias=False,
            conv2d_use_mask=True,
            linear_use_mask=False)

        print("Test Accuracy: {:.3f}".format(eval_accuracy))
        print("Classification Report:")
        print(classification_report)
        print("Global Sparsity:")
        print("{:.2f}".format(sparsity))

        # print(model.conv1._forward_pre_hooks)

        print("Fine-tuning...")

        train_model(model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device,
                    l1_regularization_strength=l1_regularization_strength,
                    l2_regularization_strength=l2_regularization_strength,
                    learning_rate=learning_rate * (learning_rate_decay**i),
                    num_epochs=num_epochs_per_iteration)

        _, eval_accuracy = evaluate_model(model=model,
                                          test_loader=test_loader,
                                          device=device,
                                          criterion=None)

        classification_report = create_classification_report(
            model=model, test_loader=test_loader, device=device)

        num_zeros, num_elements, sparsity = measure_global_sparsity(
            model,
            weight=True,
            bias=False,
            conv2d_use_mask=True,
            linear_use_mask=False)

        print("Test Accuracy: {:.3f}".format(eval_accuracy))
        print("Classification Report:")
        print(classification_report)
        print("Global Sparsity:")
        print("{:.2f}".format(sparsity))

        model_filename = "{}_{}.pt".format(model_filename_prefix, i + 1)
        model_filepath = os.path.join(model_dir, model_filename)
        save_model(model=model,
                   model_dir=model_dir,
                   model_filename=model_filename)
        model = load_model(model=model,
                           model_filepath=model_filepath,
                           device=device)

    return model


def remove_parameters(model):

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model


def main():
    parser = argparse.ArgumentParser(description='Trains resnet18 model to prepare pruning')

    parser.add_argument('--model_dir', type=str, default='saved_models', help='Path to pretrained model for pruning')
    parser.add_argument('--model_filename', type=str, default='resnet18_cifar10.pt', help='Name of pretrained model')
    parser.add_argument('--model_filename_prefix', type=str, default='pruned_model', help='Name of prefix')
    parser.add_argument('--pruned_model_filename', type=str, default='resnet18_pruned_cifar10.pt', help='Name of saving model after pruning pretrained model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs during fine tuning')
    parser.add_argument('--iters', type=int, default=5, help='Number of iterations (number of pruning)')
    parser.add_argument('--n_cpu', type=int, default=8, help='Number of cpu threads to use during batch generation')
    parser.add_argument('--train_batch_size', type=int, default=128, help='Number of batch size during training')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='Number of batch size during evaluation')
    parser.add_argument('--l1_regularization_strength', type=int, default=0, help='Number of l1 regularization strength')
    parser.add_argument('--l2_regularization_strength', type=int, default=1e-4, help='Number of l2 regularization strength')

    args = parser.parse_args()
    print(f'Command line arguments: {args}')


    num_classes = 10
    random_seed = 1
    l1_regularization_strength = args.l1_regularization_strength
    l2_regularization_strength = args.l2_regularization_strength
    learning_rate = 1e-3
    learning_rate_decay = 1

    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_dir = args.model_dir
    model_filename = args.model_filename
    model_filename_prefix = args.model_filename_prefix
    pruned_model_filename = args.pruned_model_filename
    model_filepath = os.path.join(model_dir, model_filename)
    pruned_model_filepath = os.path.join(model_dir, pruned_model_filename)

    set_random_seeds(random_seed=random_seed)

    # Create an untrained model.
    model = create_model(num_classes=num_classes)

    # Load a pretrained model.
    model = load_model(model=model,
                       model_filepath=model_filepath,
                       device=cuda_device)

    train_loader, test_loader, classes = prepare_dataloader(
        num_workers=args.n_cpu, train_batch_size=args.train_batch_sie, eval_batch_size=args.eval_batch_size)

    _, eval_accuracy = evaluate_model(model=model,
                                      test_loader=test_loader,
                                      device=cuda_device,
                                      criterion=None)

    classification_report = create_classification_report(
        model=model, test_loader=test_loader, device=cuda_device)

    num_zeros, num_elements, sparsity = measure_global_sparsity(model)

    print("Test Accuracy: {:.3f}".format(eval_accuracy))
    print("Classification Report:")
    print(classification_report)
    print("Global Sparsity:")
    print("{:.2f}".format(sparsity))

    print("Iterative Pruning + Fine-Tuning...")

    pruned_model = copy.deepcopy(model)

    # iterative_pruning_finetuning(
    #     model=pruned_model,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     device=cuda_device,
    #     learning_rate=learning_rate,
    #     learning_rate_decay=learning_rate_decay,
    #     l1_regularization_strength=l1_regularization_strength,
    #     l2_regularization_strength=l2_regularization_strength,
    #     conv2d_prune_amount=0.3,
    #     linear_prune_amount=0,
    #     num_iterations=8,
    #     num_epochs_per_iteration=50,
    #     model_filename_prefix=model_filename_prefix,
    #     model_dir=model_dir,
    #     grouped_pruning=True)

    iterative_pruning_finetuning(
        model=pruned_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=cuda_device,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        l1_regularization_strength=l1_regularization_strength,
        l2_regularization_strength=l2_regularization_strength,
        conv2d_prune_amount=0.98,
        linear_prune_amount=0,
        num_iterations=args.iters,
        num_epochs_per_iteration=args.epochs,
        model_filename_prefix=model_filename_prefix,
        model_dir=model_dir,
        grouped_pruning=True)

    # Apply mask to the parameters and remove the mask.
    remove_parameters(model=pruned_model)

    _, eval_accuracy = evaluate_model(model=pruned_model,
                                      test_loader=test_loader,
                                      device=cuda_device,
                                      criterion=None)

    classification_report = create_classification_report(
        model=pruned_model, test_loader=test_loader, device=cuda_device)

    num_zeros, num_elements, sparsity = measure_global_sparsity(pruned_model)

    print("Test Accuracy: {:.3f}".format(eval_accuracy))
    print("Classification Report:")
    print(classification_report)
    print("Global Sparsity:")
    print("{:.2f}".format(sparsity))

    save_model(model=model, model_dir=model_dir, model_filename=model_filename)


if __name__ == "__main__":

    main()
