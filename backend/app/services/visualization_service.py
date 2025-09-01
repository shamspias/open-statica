"""
Visualization Service for OpenStatica
Creates comprehensive data visualizations similar to SPSS
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional, Union
import io
import base64
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class VisualizationService:
    """Service for creating data visualizations"""

    def __init__(self):
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

        # Color palettes
        self.colors = px.colors.qualitative.Set2
        self.sequential_colors = px.colors.sequential.Viridis

    async def create_visualization(self,
                                   viz_type: str,
                                   data: pd.DataFrame,
                                   **kwargs) -> Dict[str, Any]:
        """Create a visualization based on type"""

        viz_methods = {
            'histogram': self.create_histogram,
            'boxplot': self.create_boxplot,
            'scatter': self.create_scatter_plot,
            'line': self.create_line_plot,
            'bar': self.create_bar_chart,
            'pie': self.create_pie_chart,
            'heatmap': self.create_heatmap,
            'correlation_matrix': self.create_correlation_matrix,
            'pairplot': self.create_pairplot,
            'distribution': self.create_distribution_plot,
            'qq': self.create_qq_plot,
            'residuals': self.create_residual_plots,
            'roc': self.create_roc_curve,
            'confusion_matrix': self.create_confusion_matrix_plot,
            'feature_importance': self.create_feature_importance_plot,
            'parallel_coordinates': self.create_parallel_coordinates,
            'violin': self.create_violin_plot,
            '3d_scatter': self.create_3d_scatter,
            'surface': self.create_surface_plot,
            'dendogram': self.create_dendrogram,
            'error_bars': self.create_error_bar_plot
        }

        method = viz_methods.get(viz_type)
        if not method:
            raise ValueError(f"Unknown visualization type: {viz_type}")

        return await method(data, **kwargs)

    async def create_histogram(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create histogram with normal distribution overlay"""
        column = kwargs.get('column')
        bins = kwargs.get('bins', 'auto')
        show_kde = kwargs.get('show_kde', True)
        show_normal = kwargs.get('show_normal', True)

        if column not in data.columns:
            raise ValueError(f"Column {column} not found")

        values = data[column].dropna()

        # Create figure with plotly
        fig = go.Figure()

        # Add histogram
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=30 if bins == 'auto' else bins,
            name='Frequency',
            marker_color='rgba(99, 102, 241, 0.7)',
            opacity=0.7
        ))

        # Add KDE if requested
        if show_kde:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 100)
            kde_values = kde(x_range) * len(values) * (values.max() - values.min()) / 30

            fig.add_trace(go.Scatter(
                x=x_range,
                y=kde_values,
                mode='lines',
                name='KDE',
                line=dict(color='red', width=2)
            ))

        # Add normal distribution if requested
        if show_normal:
            mean = values.mean()
            std = values.std()
            x_range = np.linspace(values.min(), values.max(), 100)
            normal_values = stats.norm.pdf(x_range, mean, std) * len(values) * (values.max() - values.min()) / 30

            fig.add_trace(go.Scatter(
                x=x_range,
                y=normal_values,
                mode='lines',
                name='Normal',
                line=dict(color='green', width=2, dash='dash')
            ))

        # Update layout
        fig.update_layout(
            title=f'Histogram of {column}',
            xaxis_title=column,
            yaxis_title='Frequency',
            bargap=0.1,
            showlegend=True,
            hovermode='x unified'
        )

        # Add statistics annotation
        stats_text = f"Mean: {values.mean():.2f}<br>"
        stats_text += f"Std: {values.std():.2f}<br>"
        stats_text += f"Skew: {values.skew():.2f}<br>"
        stats_text += f"Kurt: {values.kurtosis():.2f}"

        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.95, y=0.95,
            text=stats_text,
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )

        return {
            'type': 'histogram',
            'plotly': fig.to_dict(),
            'statistics': {
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std': float(values.std()),
                'skewness': float(values.skew()),
                'kurtosis': float(values.kurtosis()),
                'normality_test': self._test_normality(values)
            }
        }

    async def create_boxplot(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create box plot with outliers"""
        columns = kwargs.get('columns', data.select_dtypes(include=[np.number]).columns)
        group_by = kwargs.get('group_by')
        show_points = kwargs.get('show_points', False)

        fig = go.Figure()

        if group_by and group_by in data.columns:
            # Grouped boxplot
            for group in data[group_by].unique():
                group_data = data[data[group_by] == group]
                for col in columns:
                    fig.add_trace(go.Box(
                        y=group_data[col],
                        name=f'{col} ({group})',
                        boxpoints='outliers' if show_points else False,
                        marker_color=self.colors[len(fig.data) % len(self.colors)]
                    ))
        else:
            # Simple boxplot
            for col in columns:
                fig.add_trace(go.Box(
                    y=data[col],
                    name=col,
                    boxpoints='outliers' if show_points else False,
                    marker_color=self.colors[len(fig.data) % len(self.colors)],
                    boxmean='sd'  # Show mean and std
                ))

        fig.update_layout(
            title='Box Plot Analysis',
            yaxis_title='Value',
            showlegend=True,
            hovermode='x unified'
        )

        # Calculate outliers
        outliers_info = {}
        for col in columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
            outliers_info[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(data[col].dropna()) * 100
            }

        return {
            'type': 'boxplot',
            'plotly': fig.to_dict(),
            'outliers': outliers_info
        }

    async def create_scatter_plot(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create scatter plot with regression line"""
        x_col = kwargs.get('x')
        y_col = kwargs.get('y')
        color = kwargs.get('color')
        size = kwargs.get('size')
        trendline = kwargs.get('trendline', 'ols')  # ols, lowess, or None

        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color,
            size=size,
            trendline=trendline if trendline else None,
            title=f'{y_col} vs {x_col}'
        )

        # Add correlation coefficient
        if x_col and y_col:
            corr = data[[x_col, y_col]].corr().iloc[0, 1]
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.05, y=0.95,
                text=f"r = {corr:.3f}<br>r² = {corr ** 2:.3f}",
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )

        fig.update_layout(
            hovermode='closest',
            showlegend=True if color else False
        )

        return {
            'type': 'scatter',
            'plotly': fig.to_dict(),
            'correlation': float(corr) if x_col and y_col else None
        }

    async def create_correlation_matrix(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create correlation matrix heatmap"""
        method = kwargs.get('method', 'pearson')
        columns = kwargs.get('columns', data.select_dtypes(include=[np.number]).columns)

        # Calculate correlation matrix
        corr_matrix = data[columns].corr(method=method)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            reversescale=True,
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title=f'{method.capitalize()} Correlation Matrix',
            xaxis={'side': 'bottom'},
            width=700,
            height=700
        )

        # Find significant correlations
        significant_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5 and abs(corr_val) < 1:
                    significant_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })

        return {
            'type': 'heatmap',
            'plotly': fig.to_dict(),
            'significant_correlations': significant_corr
        }

    async def create_pairplot(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create pairwise scatter plot matrix"""
        columns = kwargs.get('columns', data.select_dtypes(include=[np.number]).columns[:5])
        hue = kwargs.get('hue')

        # Create scatter matrix
        fig = px.scatter_matrix(
            data,
            dimensions=columns,
            color=hue,
            title="Pairwise Relationships"
        )

        fig.update_traces(diagonal_visible=False)
        fig.update_layout(
            height=800,
            width=800
        )

        return {
            'type': 'pairplot',
            'plotly': fig.to_dict()
        }

    async def create_qq_plot(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create Q-Q plot for normality assessment"""
        column = kwargs.get('column')

        values = data[column].dropna().values

        # Calculate theoretical quantiles
        qq_data = stats.probplot(values, dist="norm")

        fig = go.Figure()

        # Add Q-Q plot points
        fig.add_trace(go.Scatter(
            x=qq_data[0][0],
            y=qq_data[0][1],
            mode='markers',
            name='Data',
            marker=dict(color='blue', size=5)
        ))

        # Add reference line
        x_range = [qq_data[0][0].min(), qq_data[0][0].max()]
        fig.add_trace(go.Scatter(
            x=x_range,
            y=[qq_data[1][1] + qq_data[1][0] * x for x in x_range],
            mode='lines',
            name='Normal',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title=f'Q-Q Plot: {column}',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles',
            showlegend=True
        )

        # Perform normality tests
        shapiro_stat, shapiro_p = stats.shapiro(values[:5000] if len(values) > 5000 else values)
        ks_stat, ks_p = stats.kstest(values, 'norm', args=(values.mean(), values.std()))

        return {
            'type': 'qq_plot',
            'plotly': fig.to_dict(),
            'normality_tests': {
                'shapiro': {'statistic': float(shapiro_stat), 'p_value': float(shapiro_p)},
                'kolmogorov_smirnov': {'statistic': float(ks_stat), 'p_value': float(ks_p)}
            }
        }

    async def create_residual_plots(self, actual: np.ndarray, predicted: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Create residual diagnostic plots"""
        residuals = actual - predicted

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals vs Fitted', 'Q-Q Plot',
                            'Scale-Location', 'Residuals vs Leverage')
        )

        # 1. Residuals vs Fitted
        fig.add_trace(go.Scatter(
            x=predicted,
            y=residuals,
            mode='markers',
            marker=dict(color='blue', size=5),
            name='Residuals'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[predicted.min(), predicted.max()],
            y=[0, 0],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Zero line'
        ), row=1, col=1)

        # 2. Q-Q plot of residuals
        qq_data = stats.probplot(residuals, dist="norm")
        fig.add_trace(go.Scatter(
            x=qq_data[0][0],
            y=qq_data[0][1],
            mode='markers',
            marker=dict(color='blue', size=5),
            name='Q-Q'
        ), row=1, col=2)

        # 3. Scale-Location (sqrt of standardized residuals)
        standardized_residuals = residuals / residuals.std()
        sqrt_abs_residuals = np.sqrt(np.abs(standardized_residuals))

        fig.add_trace(go.Scatter(
            x=predicted,
            y=sqrt_abs_residuals,
            mode='markers',
            marker=dict(color='blue', size=5),
            name='Scale-Location'
        ), row=2, col=1)

        # 4. Cook's distance (simplified)
        n = len(residuals)
        p = 1  # number of parameters (simplified)
        leverage = 1 / n + (predicted - predicted.mean()) ** 2 / ((predicted - predicted.mean()) ** 2).sum()
        cooks_d = (residuals ** 2 / (p * residuals.var())) * (leverage / (1 - leverage) ** 2)

        fig.add_trace(go.Scatter(
            x=leverage,
            y=standardized_residuals,
            mode='markers',
            marker=dict(color='blue', size=5),
            name='Leverage'
        ), row=2, col=2)

        fig.update_layout(
            title="Residual Diagnostic Plots",
            showlegend=False,
            height=800
        )

        # Diagnostic tests
        durbin_watson = self._durbin_watson(residuals)

        return {
            'type': 'residual_plots',
            'plotly': fig.to_dict(),
            'diagnostics': {
                'durbin_watson': float(durbin_watson),
                'mean_residual': float(residuals.mean()),
                'std_residual': float(residuals.std()),
                'normality_test': self._test_normality(residuals)
            }
        }

    async def create_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Create ROC curve"""
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()

        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))

        # Diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True,
            hovermode='closest'
        )

        return {
            'type': 'roc_curve',
            'plotly': fig.to_dict(),
            'auc': float(roc_auc),
            'optimal_threshold': float(thresholds[np.argmax(tpr - fpr)])
        }

    async def create_confusion_matrix_plot(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Create confusion matrix visualization"""
        from sklearn.metrics import confusion_matrix

        labels = kwargs.get('labels', sorted(np.unique(np.concatenate([y_true, y_pred]))))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=labels,
            y=labels,
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 12},
            colorscale='Blues',
            reversescale=False,
            showscale=True,
            colorbar=dict(title="Proportion")
        ))

        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            xaxis={'side': 'bottom'},
            width=500,
            height=500
        )

        return {
            'type': 'confusion_matrix',
            'plotly': fig.to_dict(),
            'matrix': cm.tolist()
        }

    async def create_feature_importance_plot(self, features: List[str], importance: List[float], **kwargs) -> Dict[
        str, Any]:
        """Create feature importance bar plot"""
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1]
        top_n = min(kwargs.get('top_n', 20), len(features))

        fig = go.Figure(go.Bar(
            x=importance[sorted_idx][:top_n],
            y=features[sorted_idx][:top_n],
            orientation='h',
            marker_color='rgba(99, 102, 241, 0.7)'
        ))

        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=max(400, top_n * 20)
        )

        return {
            'type': 'feature_importance',
            'plotly': fig.to_dict()
        }

    async def create_parallel_coordinates(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create parallel coordinates plot"""
        columns = kwargs.get('columns', data.select_dtypes(include=[np.number]).columns[:7])
        color = kwargs.get('color')

        fig = px.parallel_coordinates(
            data,
            dimensions=columns,
            color=color,
            title="Parallel Coordinates Plot"
        )

        fig.update_layout(height=500)

        return {
            'type': 'parallel_coordinates',
            'plotly': fig.to_dict()
        }

    def _test_normality(self, values: np.ndarray) -> Dict[str, Any]:
        """Test for normality"""
        shapiro_stat, shapiro_p = stats.shapiro(values[:5000] if len(values) > 5000 else values)
        return {
            'shapiro_wilk': {
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'is_normal': shapiro_p > 0.05
            }
        }

    def _durbin_watson(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic"""
        diff = np.diff(residuals)
        return np.sum(diff ** 2) / np.sum(residuals ** 2)

    # Additional visualization methods...
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
            title='Violin Plot',
            yaxis_title='Value',
            showlegend=True
        )

        return {
            'type': 'violin',
            'plotly': fig.to_dict()
        }

    async def create_3d_scatter(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create 3D scatter plot"""
        x_col = kwargs.get('x')
        y_col = kwargs.get('y')
        z_col = kwargs.get('z')
        color = kwargs.get('color')

        fig = px.scatter_3d(
            data,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color,
            title='3D Scatter Plot'
        )

        return {
            'type': '3d_scatter',
            'plotly': fig.to_dict()
        }

    async def create_surface_plot(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Create 3D surface plot"""
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])

        fig.update_layout(
            title='3D Surface Plot',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )

        return {
            'type': 'surface',
            'plotly': fig.to_dict()
        }

    async def create_dendrogram(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create dendrogram for hierarchical clustering"""
        from scipy.cluster.hierarchy import dendrogram, linkage

        columns = kwargs.get('columns', data.select_dtypes(include=[np.number]).columns)
        method = kwargs.get('method', 'ward')

        # Calculate linkage
        Z = linkage(data[columns].dropna(), method=method)

        # Create dendrogram
        fig = plt.figure(figsize=(10, 6))
        dendrogram(Z, labels=data.index)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')

        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return {
            'type': 'dendrogram',
            'image': f'data:image/png;base64,{img_base64}'
        }

    async def create_line_plot(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create line plot"""
        x_col = kwargs.get('x')
        y_cols = kwargs.get('y', [])

        if not isinstance(y_cols, list):
            y_cols = [y_cols]

        fig = go.Figure()

        for y_col in y_cols:
            fig.add_trace(go.Scatter(
                x=data[x_col] if x_col else data.index,
                y=data[y_col],
                mode='lines+markers',
                name=y_col
            ))

        fig.update_layout(
            title='Line Plot',
            xaxis_title=x_col if x_col else 'Index',
            yaxis_title='Value',
            hovermode='x unified'
        )

        return {
            'type': 'line',
            'plotly': fig.to_dict()
        }

    async def create_bar_chart(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create bar chart"""
        x_col = kwargs.get('x')
        y_col = kwargs.get('y')
        orientation = kwargs.get('orientation', 'v')

        fig = px.bar(
            data,
            x=x_col if orientation == 'v' else y_col,
            y=y_col if orientation == 'v' else x_col,
            orientation=orientation,
            title='Bar Chart'
        )

        return {
            'type': 'bar',
            'plotly': fig.to_dict()
        }

    async def create_pie_chart(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create pie chart"""
        values_col = kwargs.get('values')
        names_col = kwargs.get('names')

        fig = px.pie(
            data,
            values=values_col,
            names=names_col,
            title='Pie Chart'
        )

        return {
            'type': 'pie',
            'plotly': fig.to_dict()
        }

    async def create_distribution_plot(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create distribution comparison plot"""
        columns = kwargs.get('columns', data.select_dtypes(include=[np.number]).columns)

        fig = go.Figure()

        for col in columns:
            fig.add_trace(go.Histogram(
                x=data[col],
                name=col,
                opacity=0.7,
                nbinsx=30
            ))

        fig.update_layout(
            title='Distribution Comparison',
            xaxis_title='Value',
            yaxis_title='Frequency',
            barmode='overlay'
        )

        return {
            'type': 'distribution',
            'plotly': fig.to_dict()
        }

    async def create_error_bar_plot(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create error bar plot"""
        x_col = kwargs.get('x')
        y_col = kwargs.get('y')
        error_col = kwargs.get('error')

        fig = go.Figure(data=go.Scatter(
            x=data[x_col],
            y=data[y_col],
            error_y=dict(
                type='data',
                array=data[error_col] if error_col else data[y_col].std(),
                visible=True
            ),
            mode='markers+lines',
            marker=dict(size=8)
        ))

        fig.update_layout(
            title='Error Bar Plot',
            xaxis_title=x_col,
            yaxis_title=y_col
        )

        return {
            'type': 'error_bar',
            'plotly': fig.to_dict()
        }

    async def create_heatmap(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create generic heatmap"""
        x_col = kwargs.get('x')
        y_col = kwargs.get('y')
        z_col = kwargs.get('z')

        if z_col:
            # Pivot data for heatmap
            pivot_data = data.pivot(index=y_col, columns=x_col, values=z_col)
        else:
            # Use data directly
            pivot_data = data

        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Viridis'
        ))

        fig.update_layout(
            title='Heatmap',
            xaxis_title=x_col,
            yaxis_title=y_col
        )

        return {
            'type': 'heatmap',
            'plotly': fig.to_dict()
        }
