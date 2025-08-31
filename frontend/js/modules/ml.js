/**
 * Machine Learning Module for OpenStatica
 */

class MLModule {
    constructor(app) {
        this.app = app;
        this.currentModel = null;
        this.trainedModels = new Map();
    }

    loadMLOptions() {
        const container = document.getElementById('optionsContent');
        if (!container) return;

        container.innerHTML = `
            <div class="form-group">
                <label class="form-label">ML Task</label>
                <select class="form-control" id="mlTask">
                    <option value="classification">Classification</option>
                    <option value="regression">Regression</option>
                    <option value="clustering">Clustering</option>
                    <option value="dimensionality">Dimensionality Reduction</option>
                    <option value="anomaly">Anomaly Detection</option>
                </select>
            </div>
            
            <div id="algorithmSelection"></div>
            
            <div class="form-group">
                <label class="form-label">Target Variable</label>
                <select class="form-control" id="targetVariable">
                    <option value="">Select for supervised learning...</option>
                    ${this.getVariableOptions()}
                </select>
            </div>
            
            <div class="form-group">
                <label class="form-label">Train/Test Split</label>
                <input type="range" class="form-range" id="trainTestSplit" 
                       min="50" max="90" value="80" step="5">
                <span id="splitValue">80%</span>
            </div>
            
            <div class="form-group">
                <label class="form-checkbox">
                    <input type="checkbox" id="autoML" checked>
                    <span>Use AutoML for hyperparameter tuning</span>
                </label>
            </div>
            
            <div class="form-group">
                <label class="form-checkbox">
                    <input type="checkbox" id="crossValidate" checked>
                    <span>Perform cross-validation</span>
                </label>
            </div>
            
            <button class="btn btn-primary" onclick="window.openStatica.ml.trainModel()">
                Train Model
            </button>
            
            <div id="modelResults" class="model-results"></div>
        `;

        this.bindMLEvents();
        this.updateAlgorithmSelection('classification');
    }

    bindMLEvents() {
        document.getElementById('mlTask')?.addEventListener('change', (e) => {
            this.updateAlgorithmSelection(e.target.value);
        });

        document.getElementById('trainTestSplit')?.addEventListener('input', (e) => {
            document.getElementById('splitValue').textContent = `${e.target.value}%`;
        });
    }

    updateAlgorithmSelection(task) {
        const container = document.getElementById('algorithmSelection');
        if (!container) return;

        let algorithms = [];

        switch (task) {
            case 'classification':
                algorithms = [
                    {value: 'logistic', name: 'Logistic Regression'},
                    {value: 'svm', name: 'Support Vector Machine'},
                    {value: 'rf', name: 'Random Forest'},
                    {value: 'xgboost', name: 'XGBoost'},
                    {value: 'nn', name: 'Neural Network'},
                    {value: 'knn', name: 'K-Nearest Neighbors'},
                    {value: 'nb', name: 'Naive Bayes'},
                    {value: 'dt', name: 'Decision Tree'}
                ];
                break;

            case 'regression':
                algorithms = [
                    {value: 'linear', name: 'Linear Regression'},
                    {value: 'ridge', name: 'Ridge Regression'},
                    {value: 'lasso', name: 'Lasso Regression'},
                    {value: 'elastic', name: 'Elastic Net'},
                    {value: 'svr', name: 'Support Vector Regression'},
                    {value: 'rf_reg', name: 'Random Forest Regressor'},
                    {value: 'xgb_reg', name: 'XGBoost Regressor'},
                    {value: 'nn_reg', name: 'Neural Network Regressor'}
                ];
                break;

            case 'clustering':
                algorithms = [
                    {value: 'kmeans', name: 'K-Means'},
                    {value: 'dbscan', name: 'DBSCAN'},
                    {value: 'hierarchical', name: 'Hierarchical Clustering'},
                    {value: 'gaussian', name: 'Gaussian Mixture'},
                    {value: 'meanshift', name: 'Mean Shift'},
                    {value: 'spectral', name: 'Spectral Clustering'}
                ];
                break;

            case 'dimensionality':
                algorithms = [
                    {value: 'pca', name: 'PCA'},
                    {value: 'tsne', name: 't-SNE'},
                    {value: 'umap', name: 'UMAP'},
                    {value: 'lda', name: 'LDA'},
                    {value: 'ica', name: 'ICA'},
                    {value: 'autoencoder', name: 'Autoencoder'}
                ];
                break;

            case 'anomaly':
                algorithms = [
                    {value: 'isolation', name: 'Isolation Forest'},
                    {value: 'lof', name: 'Local Outlier Factor'},
                    {value: 'ocsvm', name: 'One-Class SVM'},
                    {value: 'elliptic', name: 'Elliptic Envelope'},
                    {value: 'autoencoder_ad', name: 'Autoencoder'}
                ];
                break;
        }

        container.innerHTML = `
            <div class="form-group">
                <label class="form-label">Algorithm</label>
                <select class="form-control" id="mlAlgorithm">
                    ${algorithms.map(alg =>
            `<option value="${alg.value}">${alg.name}</option>`
        ).join('')}
                </select>
            </div>
        `;
    }

    getVariableOptions() {
        if (!this.app.data) return '';

        return [...this.app.data.numeric_columns, ...this.app.data.categorical_columns]
            .map(col => `<option value="${col}">${col}</option>`)
            .join('');
    }

    async trainModel() {
        try {
            this.app.showLoading(true);

            const task = document.getElementById('mlTask')?.value;
            const algorithm = document.getElementById('mlAlgorithm')?.value;
            const target = document.getElementById('targetVariable')?.value;
            const features = this.app.getSelectedVariables();
            const trainTestSplit = document.getElementById('trainTestSplit')?.value / 100;
            const autoML = document.getElementById('autoML')?.checked;
            const crossValidate = document.getElementById('crossValidate')?.checked;

            if (features.length === 0) {
                throw new Error('Please select feature variables');
            }

            if (['classification', 'regression'].includes(task) && !target) {
                throw new Error('Please select a target variable');
            }

            const params = {
                task: task,
                algorithm: algorithm,
                features: features,
                target: target,
                train_test_split: trainTestSplit,
                auto_ml: autoML,
                cross_validate: crossValidate
            };

            const result = await this.app.api.trainModel(this.app.sessionId, params);

            // Store trained model
            this.currentModel = result.model_id;
            this.trainedModels.set(result.model_id, result);

            // Display results
            this.displayModelResults(result);

            this.app.showNotification('Model trained successfully!', 'success');

        } catch (error) {
            this.app.showNotification(error.message, 'error');
        } finally {
            this.app.showLoading(false);
        }
    }

    displayModelResults(results) {
        const container = document.getElementById('modelResults');
        if (!container) return;

        container.innerHTML = `
            <div class="model-card">
                <h3>Model Performance</h3>
                
                <div class="model-metrics">
                    ${this.formatModelMetrics(results.metrics)}
                </div>
                
                ${results.feature_importance ? `
                    <div class="feature-importance">
                        <h4>Feature Importance</h4>
                        ${this.formatFeatureImportance(results.feature_importance)}
                    </div>
                ` : ''}
                
                ${results.confusion_matrix ? `
                    <div class="confusion-matrix">
                        <h4>Confusion Matrix</h4>
                        ${this.formatConfusionMatrix(results.confusion_matrix)}
                    </div>
                ` : ''}
                
                <div class="model-actions">
                    <button class="btn btn-secondary" onclick="window.openStatica.ml.downloadModel('${results.model_id}')">
                        Download Model
                    </button>
                    <button class="btn btn-primary" onclick="window.openStatica.ml.deployModel('${results.model_id}')">
                        Deploy Model
                    </button>
                </div>
            </div>
        `;
    }

    formatModelMetrics(metrics) {
        let html = '<div class="metrics-grid">';

        for (const [key, value] of Object.entries(metrics)) {
            html += `
                <div class="metric-item">
                    <div class="metric-label">${this.formatMetricName(key)}</div>
                    <div class="metric-value">${this.formatMetricValue(value)}</div>
                </div>
            `;
        }

        html += '</div>';
        return html;
    }

    formatMetricName(name) {
        const nameMap = {
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_score': 'F1 Score',
            'auc_roc': 'AUC-ROC',
            'mse': 'Mean Squared Error',
            'rmse': 'Root MSE',
            'mae': 'Mean Absolute Error',
            'r2': 'R² Score'
        };

        return nameMap[name] || name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    formatMetricValue(value) {
        if (typeof value === 'number') {
            return value.toFixed(4);
        }
        return value;
    }

    formatFeatureImportance(importance) {
        const sorted = Object.entries(importance).sort((a, b) => b[1] - a[1]);

        let html = '<div class="importance-chart">';

        sorted.slice(0, 10).forEach(([feature, value]) => {
            const percentage = (value * 100).toFixed(1);
            html += `
                <div class="importance-bar">
                    <div class="importance-label">${feature}</div>
                    <div class="importance-value">
                        <div class="importance-fill" style="width: ${percentage}%"></div>
                        <span>${percentage}%</span>
                    </div>
                </div>
            `;
        });

        html += '</div>';
        return html;
    }

    formatConfusionMatrix(matrix) {
        // Simple 2x2 confusion matrix for binary classification
        if (matrix.length === 2) {
            return `
                <table class="confusion-matrix-table">
                    <thead>
                        <tr>
                            <th></th>
                            <th>Predicted 0</th>
                            <th>Predicted 1</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <th>Actual 0</th>
                            <td>${matrix[0][0]}</td>
                            <td>${matrix[0][1]}</td>
                        </tr>
                        <tr>
                            <th>Actual 1</th>
                            <td>${matrix[1][0]}</td>
                            <td>${matrix[1][1]}</td>
                        </tr>
                    </tbody>
                </table>
            `;
        }

        // For multi-class, create a more complex table
        return '<div>Multi-class confusion matrix visualization</div>';
    }

    async downloadModel(modelId) {
        try {
            const blob = await this.app.api.exportResults(this.app.sessionId, 'model');
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `model_${modelId}.pkl`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (error) {
            this.app.showNotification('Failed to download model', 'error');
        }
    }

    async deployModel(modelId) {
        // TODO: Implement model deployment
        this.app.showNotification('Model deployment coming soon!', 'info');
    }

    async runMLAnalysis(variables) {
        // This is called from the main app
        return await this.trainModel();
    }
}