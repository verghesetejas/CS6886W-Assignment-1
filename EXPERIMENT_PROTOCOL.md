# Experiment protocol

- Activation study: compare ReLU, Sigmoid, Tanh, SiLU and GELU while holding optimizer, learning rate, batch size, epoch count, data split and seed fixed.
- Optimizer study: compare SGD, Nesterov-SGD, Adam, Adagrad, RMSprop and Nadam while holding other settings fixed. Record momentum and weight decay.
- Hyperparameter study: vary batch size, learning rate and epoch count. Use controlled comparisons to distinguish their effects.
- Select exactly one best run using validation accuracy. Re-run it with the same seed and settings. Only then evaluate once on the official CIFAR-10 test set.
- Report measured values only. Record run IDs, settings, package versions, GPU, seed, failed runs and any deviations.
