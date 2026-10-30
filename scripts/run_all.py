"""Backward-compatible entry point: run CPU-friendly classical models only.

Deep learning (LSTM/CNN/Transformer) is intentionally delegated to
notebooks/colab_deep_learning.ipynb so the local PC does not need a GPU.
"""
from run_classical_cpu import main

if __name__ == '__main__':
    main()
