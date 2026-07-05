#!/bin/bash
cd "$(dirname "$0")"
bash engine-scripts/build_leela.sh
bash engine-scripts/build_stockfish.sh
