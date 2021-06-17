import os
import torch
from utils import set_random_seeds, create_model, prepare_dataloader, train_model, save_model, load_model, evaluate_model, create_classification_report

import argparse


def main():
    parser = argparse.ArgumentParser(description='Trains resnet18 model to prepare pruning')

    parser.add_argument('--model_dir', type=str, default='saved_models', help='Path to model saving after training')
    parser.add_argument('--model_filename', type=str, default='resnet18_cifar10.pt', help='Name of saving model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--n_cpu', type=int, default=8, help='Number of cpu threads to use during batch generation')
    parser.add_argument('--train_batch_size', type=int, default=128, help='Number of batch size during training')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='Number of batch size during evaluation')
    parser.add_argument('--l1_regularization_strength', type=int, default=0, help='Number of l1 regularization strength')
    parser.add_argument('--l2_regularization_strength', type=int, default=1e-4, help='Number of l2 regularization strength')

    args = parser.parse_args()
    print(f'Command line arguments: {args}')
    
    random_seed = 0
    num_classes = 10
    l1_regularization_strength = args.l1_regularization_strength
    l2_regularization_strength = args.l2_regularization_strength
    learning_rate = 1e-1
    num_epochs = args.epochs
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_dir = args.model_dir
    model_filename = args.model_filename
    model_filepath = os.path.join(model_dir, model_filename)

    set_random_seeds(random_seed=random_seed)

    # Create an untrained model.
    model = create_model(num_classes=num_classes)

    train_loader, test_loader, classes = prepare_dataloader(
        num_workers=args.n_cpu, train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size)

    # Train model.
    print("Training Model...")
    model = train_model(model=model,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        device=cuda_device,
                        l1_regularization_strength=l1_regularization_strength,
                        l2_regularization_strength=l2_regularization_strength,
                        learning_rate=learning_rate,
                        num_epochs=num_epochs)

    # Save model.
    save_model(model=model, model_dir=model_dir, model_filename=model_filename)
    # Load a pretrained model.
    model = load_model(model=model,
                       model_filepath=model_filepath,
                       device=cuda_device)

    _, eval_accuracy = evaluate_model(model=model,
                                      test_loader=test_loader,
                                      device=cuda_device,
                                      criterion=None)

    classification_report = create_classification_report(
        model=model, test_loader=test_loader, device=cuda_device)

    print("Test Accuracy: {:.3f}".format(eval_accuracy))
    print("Classification Report:")
    print(classification_report)


if __name__ == "__main__":

    main()
