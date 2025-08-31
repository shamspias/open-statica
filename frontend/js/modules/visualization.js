/**
 * Visualization Module for OpenStatica
 */

class VisualizationModule {
    constructor(app) {
        this.app = app;
        this.charts = new Map();
        this.currentChart = null;
    }

    async createVisualizations(results) {
        const container = document.getElementById('vizContainer');
        if (!container) return;

        // Clear previous visualizations
        container.innerHTML = '';

        // Determine visualization type based on results
        if (results.visualization_type) {
            await this.createSpecificVisualization(results, container);
        } else {
            // Auto-detect appropriate visualizations
            await this.autoCreateVisualizations(results, container);
        }
    }

    async autoCreateVisualizations(results, container) {
        // Create appropriate visualizations based on data type
        if (results.histogram_data) {
            await this.createHistogram(results.histogram_data, container);
        }

        if (results.scatter_data) {
            await this.createScatterPlot(results.scatter_data, container);
        }

        if (results.box_plot_data) {
            await this.createBoxPlot(results.box_plot_data, container);
        }

        if (results.correlation_matrix) {
            await this.createHeatmap(results.correlation_matrix, container);
        }

        if (results.time_series_data) {
            await this.createTimeSeriesPlot(results.time_series_data, container);
        }
    }

    async createHistogram(data, container) {
        const div = document.createElement('div');
        div.className = 'chart-container';
        div.innerHTML = '<canvas id="histogramChart"></canvas>';
        container.appendChild(div);

        const ctx = document.getElementById('histogramChart').getContext('2d');

        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.bins,
                datasets: [{
                    label: data.label || 'Frequency',
                    data: data.frequencies,
                    backgroundColor: 'rgba(99, 102, 241, 0.5)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: data.title || 'Histogram'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: data.xlabel || 'Value'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Frequency'
                        }
                    }
                }
            }
        });

        this.charts.set('histogram', chart);
    }

    async createScatterPlot(data, container) {
        const div = document.createElement('div');
        div.className = 'chart-container';
        div.id = 'scatterPlot';
        container.appendChild(div);

        const trace = {
            x: data.x,
            y: data.y,
            mode: 'markers',
            type: 'scatter',
            name: data.label || 'Data',
            marker: {
                color: data.color || 'rgba(99, 102, 241, 0.7)',
                size: 8
            }
        };

        const layout = {
            title: data.title || 'Scatter Plot',
            xaxis: {title: data.xlabel || 'X'},
            yaxis: {title: data.ylabel || 'Y'},
            height: 400
        };

        // Add regression line if available
        if (data.regression_line) {
            const lineTrace = {
                x: data.regression_line.x,
                y: data.regression_line.y,
                mode: 'lines',
                type: 'scatter',
                name: 'Regression Line',
                line: {
                    color: 'rgba(239, 68, 68, 1)',
                    width: 2
                }
            };
            Plotly.newPlot('scatterPlot', [trace, lineTrace], layout);
        } else {
            Plotly.newPlot('scatterPlot', [trace], layout);
        }
    }

    async createBoxPlot(data, container) {
        const div = document.createElement('div');
        div.className = 'chart-container';
        div.id = 'boxPlot';
        container.appendChild(div);

        const traces = data.groups.map((group, i) => ({
            y: group.values,
            type: 'box',
            name: group.name,
            marker: {
                color: this.getColor(i)
            }
        }));

        const layout = {
            title: data.title || 'Box Plot',
            yaxis: {title: data.ylabel || 'Value'},
            height: 400
        };

        Plotly.newPlot('boxPlot', traces, layout);
    }

    async createHeatmap(data, container) {
        const div = document.createElement('div');
        div.className = 'chart-container';
        div.id = 'heatmap';
        container.appendChild(div);

        const trace = {
            z: data.values,
            x: data.columns,
            y: data.rows,
            type: 'heatmap',
            colorscale: 'RdBu',
            reversescale: true,
            showscale: true
        };

        const layout = {
            title: data.title || 'Correlation Heatmap',
            height: 500,
            width: 600,
            annotations: []
        };

        // Add text annotations
        for (let i = 0; i < data.rows.length; i++) {
            for (let j = 0; j < data.columns.length; j++) {
                layout.annotations.push({
                    x: data.columns[j],
                    y: data.rows[i],
                    text: data.values[i][j].toFixed(2),
                    showarrow: false,
                    font: {color: Math.abs(data.values[i][j]) > 0.5 ? 'white' : 'black'}
                });
            }
        }

        Plotly.newPlot('heatmap', [trace], layout);
    }

    async createTimeSeriesPlot(data, container) {
        const div = document.createElement('div');
        div.className = 'chart-container';
        div.id = 'timeSeriesPlot';
        container.appendChild(div);

        const traces = data.series.map((series, i) => ({
            x: series.dates,
            y: series.values,
            type: 'scatter',
            mode: 'lines',
            name: series.name,
            line: {
                color: this.getColor(i),
                width: 2
            }
        }));

        const layout = {
            title: data.title || 'Time Series',
            xaxis: {
                title: 'Date',
                type: 'date'
            },
            yaxis: {
                title: data.ylabel || 'Value'
            },
            height: 400
        };

        Plotly.newPlot('timeSeriesPlot', traces, layout);
    }

    async create3DPlot(data, container) {
        const div = document.createElement('div');
        div.className = 'chart-container';
        div.id = 'plot3D';
        container.appendChild(div);

        const trace = {
            x: data.x,
            y: data.y,
            z: data.z,
            mode: 'markers',
            type: 'scatter3d',
            marker: {
                color: data.color || data.z,
                colorscale: 'Viridis',
                showscale: true,
                size: 5
            }
        };

        const layout = {
            title: data.title || '3D Scatter Plot',
            scene: {
                xaxis: {title: data.xlabel || 'X'},
                yaxis: {title: data.ylabel || 'Y'},
                zaxis: {title: data.zlabel || 'Z'}
            },
            height: 500
        };

        Plotly.newPlot('plot3D', [trace], layout);
    }

    async createNetworkGraph(data, container) {
        // Network visualization for clustering or graph analysis
        const div = document.createElement('div');
        div.className = 'chart-container';
        div.id = 'networkGraph';
        container.appendChild(div);

        // TODO: Implement network graph using D3.js or vis.js
        div.innerHTML = '<p>Network graph visualization coming soon</p>';
    }

    getColor(index) {
        const colors = [
            '#6366f1', '#8b5cf6', '#ec4899', '#f43f5e',
            '#f97316', '#eab308', '#84cc16', '#22c55e',
            '#10b981', '#14b8a6', '#06b6d4', '#0ea5e9'
        ];
        return colors[index % colors.length];
    }

    async exportChart(format = 'png') {
        if (!this.currentChart) {
            this.app.showNotification('No chart to export', 'warning');
            return;
        }

        try {
            if (format === 'png') {
                const canvas = document.querySelector('.chart-container canvas');
                if (canvas) {
                    const url = canvas.toDataURL('image/png');
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'chart.png';
                    a.click();
                }
            } else if (format === 'svg') {
                // For Plotly charts
                Plotly.downloadImage(this.currentChart, {
                    format: 'svg',
                    filename: 'chart'
                });
            }
        } catch (error) {
            this.app.showNotification('Failed to export chart', 'error');
        }
    }

    clearCharts() {
        this.charts.forEach(chart => {
            if (chart.destroy) {
                chart.destroy();
            }
        });
        this.charts.clear();

        const container = document.getElementById('vizContainer');
        if (container) {
            container.innerHTML = '';
        }
    }
}