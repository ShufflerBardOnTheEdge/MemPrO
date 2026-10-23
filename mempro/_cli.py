"""
Entry point for the mempro console script.

NUM_CPU must be set before MemPrO is imported because XLA reads it at
module level to configure the fake device count for pmap. This shim
pre-parses -nc before the import happens.
"""

import os
import argparse


def main():
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("-nc", "--num_cpus", type=int, default=None)
    _pre_args, _ = _pre.parse_known_args()

    if _pre_args.num_cpus is not None:
        os.environ["NUM_CPU"] = str(_pre_args.num_cpus)

    # Import deferred until NUM_CPU is set above
    from mempro import MemPrO
    MemPrO.main()
