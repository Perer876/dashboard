import pandas as pd
import plotly.graph_objects as go


def to_scatter3d(path: pd.DataFrame) -> go.Scatter3d:
    """Convert a path to a Scatter3d plot"""
    return go.Scatter3d(
        x=path.values[:, 0],
        y=path.values[:, 1],
        z=path.index,
        marker=dict(size=2),
        line=dict(
            width=2,
            color=path.index,
            colorscale="Viridis",
            colorbar=dict(title="Time"),
        ),
        mode="lines",
        showlegend=True,
    )
