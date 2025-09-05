import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any
import json


class VisualizationService:
    """Simplified visualization service"""

    async def create_chart(self, df: pd.DataFrame, chart_type: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create various chart types"""

        if chart_type == 'histogram':
            return await self.histogram(df, request)
        elif chart_type == 'scatter':
            return await self.scatter_plot(df, request)
        elif chart_type == 'box':
            return await self.box_plot(df, request)
        elif chart_type == 'bar':
            return await self.bar_chart(df, request)
        elif chart_type == 'line':
            return await self.line_chart(df, request)
        elif chart_type == 'correlation_heatmap':
            return await self.correlation_heatmap(df, request)
        else:
            return {'error': 'Unknown chart type'}

    async def histogram(self, df: pd.DataFrame, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create histogram"""
        column = request.get('column')

        if column not in df.columns:
            return {'error': f'Column {column} not found'}

        fig = go.Figure(data=[
            go.Histogram(x=df[column].dropna().tolist(), nbinsx=30)
        ])

        fig.update_layout(
            title=f'Distribution of {column}',
            xaxis_title=column,
            yaxis_title='Frequency'
        )

        return {'chart': json.loads(fig.to_json())}

    async def scatter_plot(self, df: pd.DataFrame, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create scatter plot"""
        x_col = request.get('x')
        y_col = request.get('y')

        if x_col not in df.columns or y_col not in df.columns:
            return {'error': 'Invalid column selection'}

        fig = go.Figure(data=[
            go.Scatter(
                x=df[x_col].tolist(),
                y=df[y_col].tolist(),
                mode='markers'
            )
        ])

        fig.update_layout(
            title=f'{y_col} vs {x_col}',
            xaxis_title=x_col,
            yaxis_title=y_col
        )

        # Add correlation
        try:
            corr = df[[x_col, y_col]].corr().iloc[0, 1]
            fig.add_annotation(
                x=0.05, y=0.95,
                text=f"r = {corr:.3f}",
                showarrow=False,
                xref="paper", yref="paper"
            )
        except:
            pass

        return {'chart': json.loads(fig.to_json())}

    async def box_plot(self, df: pd.DataFrame, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create box plot"""
        columns = request.get('columns', [])

        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        fig = go.Figure()
        for col in columns:
            if col in df.columns:
                fig.add_trace(go.Box(y=df[col].tolist(), name=col))

        fig.update_layout(title='Box Plot', yaxis_title='Value')

        return {'chart': json.loads(fig.to_json())}

    async def bar_chart(self, df: pd.DataFrame, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create bar chart"""
        x_col = request.get('x')
        y_col = request.get('y')

        if x_col not in df.columns or y_col not in df.columns:
            return {'error': 'Invalid column selection'}

        # If categorical x, aggregate
        if df[x_col].dtype == 'object':
            data = df.groupby(x_col)[y_col].mean().reset_index()
        else:
            data = df[[x_col, y_col]]

        fig = go.Figure(data=[
            go.Bar(
                x=data[x_col].tolist(),
                y=data[y_col].tolist()
            )
        ])

        fig.update_layout(
            title=f'{y_col} by {x_col}',
            xaxis_title=x_col,
            yaxis_title=y_col
        )

        return {'chart': json.loads(fig.to_json())}

    async def line_chart(self, df: pd.DataFrame, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create line chart"""
        x_col = request.get('x')
        y_cols = request.get('y', [])

        if not isinstance(y_cols, list):
            y_cols = [y_cols]

        fig = go.Figure()

        for y_col in y_cols:
            if y_col in df.columns:
                x_data = df[x_col].tolist() if x_col and x_col in df.columns else list(range(len(df)))
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=df[y_col].tolist(),
                    mode='lines+markers',
                    name=y_col
                ))

        fig.update_layout(
            title='Line Chart',
            xaxis_title=x_col if x_col else 'Index',
            yaxis_title='Value'
        )

        return {'chart': json.loads(fig.to_json())}

    async def correlation_heatmap(self, df: pd.DataFrame, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create correlation heatmap"""
        columns = request.get('columns', [])

        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(columns) < 2:
            return {'error': 'Need at least 2 numeric columns for correlation'}

        # Calculate correlation matrix
        corr_matrix = df[columns].corr()

        # Convert to proper format for JSON serialization
        z_values = corr_matrix.values.tolist()
        x_labels = corr_matrix.columns.tolist()
        y_labels = corr_matrix.index.tolist()

        # Create text annotations
        text_values = [[f'{val:.2f}' for val in row] for row in z_values]

        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=text_values,
            texttemplate='%{text}',
            colorbar=dict(title="Correlation"),
            reversescale=True
        ))

        fig.update_layout(
            title='Correlation Matrix',
            width=600,
            height=600,
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'}
        )

        # Convert to JSON and parse it to ensure it's serializable
        return {'chart': json.loads(fig.to_json())}
