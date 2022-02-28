Code for experiment 4.2 of the paper.

## Requirements
torch, numpy, wandb

## Usage

Run the main.py file with the following arguments:

* ```dataset='mnist'``` to choose  MNIST and ```dataset='deepsat'``` for SAT-6 
* ```-Q=0``` to run the classical architecture and ```-Q=1``` to run the quantum-classical architecture 
* ```-seed``` for setting the seed of the computation. The 10 runs of the paper used seeds (0, ..., 9) for both the classical and the hybrid architectures.

## Example

```python main.py -dataset='mnist' -Q=0 -seed=0```



