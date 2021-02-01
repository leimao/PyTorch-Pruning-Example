# PyTorch Pruning

## Introduction

PyTorch pruning example for ResNet.

## Usages

### Build Docker Image

```
$ docker build -f docker/pytorch.Dockerfile --no-cache --tag=pytorch:1.7.1 .
```

### Run Docker Container

```
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt pytorch:1.7.1
```

### Run Pre-Training

```
$ python pretrain.py
```

### Run Pruning

```
$ python prune.py
```

## References

* [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)