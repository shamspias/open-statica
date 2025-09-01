import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional
import io
import base64


class VisualizationService:
    """Service for creating data visualizations"""

    def __init__(self):
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)

    async def create_histogram(self, data: pd.Series, **kwargs) -> Dict[str, Any]:
        """Create histogram"""
        bins = kwargs.get('bins', 'auto')
        title = kwargs.get('title', f'Histogram of {data.name}')

        # Calculate histogram data
        counts, edges = np.histogram(data.dropna(), bins=bins)

        # Create plotly figure
        fig = go.Figure(data=[
            go.Bar(
                x=edges[:-1],
                y=counts,
                width=edges[1] - edges[0],
                marker_color='rgba(99, 102, 241, 0.7)',
                marker_line_color='rgba(99, 102, 241, 1)',
                marker_line_width=1
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title=data.name,
            yaxis_title='Frequency',
            showlegend=False,
            hovermode='x'
        )

        return {
            "type": "histogram",
            "data": {
                "bins": edges.tolist(),
                "frequencies": counts.tolist(),
                "title": title,
                "xlabel": data.name
            },
            "plotly": fig.to_dict()
        }

    async def create_scatter_plot(self, x: pd.Series, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Create scatter plot"""
        title = kwargs.get('title', f'{y.name} vs {x.name}')
        color = kwargs.get('color', None)
        size = kwargs.get('size', None)

        # Create plotly figure
        fig = px.scatter(
            x=x, y=y,
            color=color,
            size=size,
            title=title,
            labels={
                'x': x.name,
                'y': y.name
            }
        )

        # Add regression line if requested
        if kwargs.get('trendline', False):
            from scipy import stats
            slope, intercept, r_value, _, _ = stats.linregress(x.dropna(), y.dropna())
            x_line = np.array([x.min(), x.max()])
            y_line = slope * x_line + intercept

            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                name=f'R² = {r_value ** 2:.3f}',
                line=dict(color='red', width=2)
            ))

        return {
            "type": "scatter",
            "data": {
                "x": x.tolist(),
                "y": y.tolist(),
                "title": title,
                "xlabel": x.name,
                "ylabel": y.name
            },
            "plotly": fig.to_dict()
        }

    async def create_box_plot(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create box plot"""
        columns = kwargs.get('columns', data.select_dtypes(include=[np.number]).columns)
        title = kwargs.get('title', 'Box Plot')

        # Create plotly figure
        fig = go.Figure()

        for col in columns:
            fig.add_trace(go.Box(
                y=data[col],
                name=col,
                boxmean='sd'
            ))

        fig.update_layout(
            title=title,
            yaxis_title='Value',
            showlegend=True
        )

        return {
            "type": "boxplot",
            "data": {
                "groups": [
                    {"name": col, "values": data[col].dropna().tolist()}
                    for col in columns
                ],
                "title": title
            },
            "plotly": fig.to_dict()
        }

    async def create_heatmap(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create correlation heatmap"""
        # Calculate correlation matrix
        corr_matrix = data.corr()

        # Create plotly figure
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            reversescale=True
        ))

        fig.update_layout(
            title=kwargs.get('title', 'Correlation Heatmap'),
            xaxis={'side': 'bottom'},
            width=600,
            height=600
        )

        return {
            "type": "heatmap",
            "data": {
                "values": corr_matrix.values.tolist(),
                "columns": corr_matrix.columns.tolist(),
                "rows": corr_matrix.columns.tolist(),
                "title": kwargs.get('title', 'Correlation Heatmap')
            },
            "plotly": fig.to_dict()
        }

    async def create_time_series(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create time series plot"""
        date_column = kwargs.get('date_column')
        value_columns = kwargs.get('value_columns', data.select_dtypes(include=[np.number]).columns)

        # Create plotly figure
        fig = go.Figure()

        for col in value_columns:
            fig.add_trace(go.Scatter(
                x=data[date_column] if date_column else data.index,
                y=data[col],
                mode='lines',
                name=col
            ))

        fig.update_layout(
            title=kwargs.get('title', 'Time Series'),
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified'
        )

        return {
            "type": "timeseries",
            "data": {
                "series": [
                    {
                        "name": col,
                        "dates": (data[date_column] if date_column else data.index).tolist(),
                        "values": data[col].tolist()
                    }
                    for col in value_columns
                ],
                "title": kwargs.get('title', 'Time Series')
            },
            "plotly": fig.to_dict()
        }

    async def create_3d_scatter(self, x: pd.Series, y: pd.Series, z: pd.Series, **kwargs) -> Dict[str, Any]:
        """Create 3D scatter plot"""
        color = kwargs.get('color', z)

        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Color Scale")
            )
        )])

        fig.update_layout(
            title=kwargs.get('title', '3D Scatter Plot'),
            scene=dict(
                xaxis_title=x.name,
                yaxis_title=y.name,
                zaxis_title=z.name
            )
        )

        return {
            "type": "scatter3d",
            "data": {
                "x": x.tolist(),
                "y": y.tolist(),
                "z": z.tolist(),
                "title": kwargs.get('title', '3D Scatter Plot')
            },
            "plotly": fig.to_dict()
        }

    async def create_pie_chart(self, data: pd.Series, **kwargs) -> Dict[str, Any]:
        """Create pie chart"""
        fig = px.pie(
            values=data.values,
            names=data.index,
            title=kwargs.get('title', f'Distribution of {data.name}')
        )

        return {
            "type": "pie",
            "plotly": fig.to_dict()
        }

    async def create_violin_plot(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create violin plot"""
        columns = kwargs.get('columns', data.select_dtypes(include=[np.number]).columns)

        fig = go.Figure()

        for col in columns:
            fig.add_trace(go.Violin(
                y=data[col],
                name=col,
                box_visible=True,
                meanline_visible=True
            ))

        fig.update_layout(
            title=kwargs.get('title', 'Violin Plot'),
            yaxis_title='Value'
        )

        return {
            "type": "violin",
            "plotly": fig.to_dict()
        }

    def save_figure_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
