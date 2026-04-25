#!/bin/sh

echo "Setting up environment..."

echo "Set up dev tools..."
{
    pre-commit install
}

echo "check uv installation..."
{
    uv python list
    uv run --python 3.14 python -c \
    "import sys; print('Python 3.14 GIL enabled:', sys._is_gil_enabled())"

    uv run --python 3.14+freethreaded python -c \
    "import sys; print('Python 3.14t GIL enabled:', sys._is_gil_enabled())"
}

echo "Setup complete"
