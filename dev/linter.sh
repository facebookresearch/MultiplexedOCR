#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates.

# Run this script at project root by "bash ./dev/linter.sh" before you commit

{
  black --version | grep -E "21\." > /dev/null
} || {
  echo "Linter requires 'black==21.*' !"
  exit 1
}

ISORT_VERSION=$(isort --version-number)
if [[ "$ISORT_VERSION" != 5.9* ]]; then
  echo "Linter requires isort==5.9.3 !"
  exit 1
fi

set -v

echo "Running isort ..."
isort . --atomic

echo "Running black ..."
black -l 100 .

echo "Running flake8 ..."
if [ -x "$(command -v flake8-3)" ]; then
  flake8-3 .
else
  # python3 -m flake8 . --exclude .ipynb_checkpoints
  python3 -m flake8 .
fi

# echo "Running clang-format ..."
# find . -regex ".*\.\(cpp\|c\|cc\|cu\|cxx\|h\|hh\|hpp\|hxx\|tcc\|mm\|m\)" -print0 | xargs -0 clang-format -i

# command -v arc > /dev/null && arc lint
