#!/usr/bin/env python3
"""Per-neuron SNN certificate sweep on fashion_mnist. See deep_core.py."""
from deep_core import build_argparser, run_experiment
DATASET = "fashion_mnist"
if __name__ == "__main__":
    run_experiment(DATASET, build_argparser(DATASET).parse_args())
