# Dashboard

## Requirements

You should have the [uv](https://docs.astral.sh/uv/getting-started/installation/) tool installed on your system.
This tool is used to manage the virtual environment and run the Jupyter Lab server for us.

## Installation

Clone the repository and install the dependencies:

```bash
git clone git@github.com:Perer876/dashboard.git
cd random-walks
```

## Usage

Install the virtual environment as a kernel in the Jupyter Lab server.
You can later select the kernel from the Jupyter Lab server.

```bash
uv run ipython kernel install --user --env VIRTUAL_ENV .venv --name=project
```

To start the Jupyter Lab server, run the following command:

```bash
uv run --with jupyter jupyter lab
```

The Jupyter Lab server will start on [http://localhost:8888](http://localhost:8888).

## Development

### Formatting

To format the code, run the following command:

```bash
uvx ruff format
```
