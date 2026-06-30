#!/usr/bin/env python3
"""Per-neuron SNN certificate sweep on cifar10. See deep_core.py."""
from deep_core import build_argparser, run_experiment
DATASET = "cifar10"
if __name__ == "__main__":
    run_experiment(DATASET, build_argparser(DATASET).parse_args())
