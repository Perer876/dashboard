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

To display the dashboard, run the following command:

```bash
uv run panel serve main.ipynb --dev
```

## Development

### Formatting

To format the code, run the following command:

```bash
uvx ruff format
```
