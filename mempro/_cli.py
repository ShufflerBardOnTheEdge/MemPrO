"""
Entry point for the mempro console script.

Device options (-nc, -dev) are read when MemPrO is imported, because JAX has
to be configured before it starts; see _pre_parse_device_args in MemPrO.py.
"""


def main():
    from mempro import MemPrO
    MemPrO.main()
