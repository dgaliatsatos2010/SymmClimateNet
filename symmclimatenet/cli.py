from __future__ import annotations
import argparse, json
from . import __version__
from .model import locked_spec_dict

def main():
    p=argparse.ArgumentParser(prog="symmclimatenet", description="SymmClimateNet utilities")
    sub=p.add_subparsers(dest="cmd")
    sub.add_parser("info", help="Print release and locked-architecture metadata")
    args=p.parse_args()
    if args.cmd in (None, "info"):
        print(json.dumps({"name":"SymmClimateNet","version":__version__,"locked_architecture":locked_spec_dict()}, indent=2))

if __name__ == "__main__":
    main()
