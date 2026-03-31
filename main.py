#!/usr/bin/env python3
"""CLI entrypoint for modular triplet ML pipeline stages."""

from __future__ import annotations

import argparse

import dataset_build
import dataset_prepare
import infer
import plotting
import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Triplet-level ML pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_build.register_subparser(subparsers)
    dataset_prepare.register_subparser(subparsers)
    train.register_subparser(subparsers)
    infer.register_subparser(subparsers)
    plotting.register_plot_subparsers(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
