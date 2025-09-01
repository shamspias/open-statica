/**
 * Data Module for OpenStatica
 */

class DataModule {
    constructor(app) {
        this.app = app;
        this.currentData = null;
        this.transformations = [];
    }

    displayData(dataInfo) {
        this.currentData = dataInfo;

        // Update file info
        this.updateFileInfo(dataInfo);

        // Display data preview
        this.displayDataTable(dataInfo.preview);

        // Show data statistics
        this.displayDataStatistics(dataInfo);

        // Update variable list
        this.updateVariableList(dataInfo);
    }

    updateFileInfo(dataInfo) {
        const fileInfoElement = document.querySelector('.file-name');
        if (fileInfoElement) {
            fileInfoElement.textContent = `${dataInfo.rows} rows × ${dataInfo.columns} columns`;
        }
    }

    displayDataTable(preview) {
        const container = document.getElementById('dataTableContainer');
        if (!container || !preview || preview.length === 0) return;

        const columns = Object.keys(preview[0]);

        let html = `
            <div class="data-table-wrapper">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th class="row-number">#</th>
                            ${columns.map(col => `
                                <th>
                                    <div class="column-header">
                                        <span class="column-name">${col}</span>
                                        <span class="column-type">${this.getColumnType(col)}</span>
                                    </div>
                                </th>
                            `).join('')}
                        </tr>
                    </thead>
                    <tbody>
        `;

        preview.forEach((row, index) => {
            html += `
                <tr>
                    <td class="row-number">${index + 1}</td>
                    ${columns.map(col => `
                        <td class="${this.getCellClass(row[col])}">
                            ${this.formatCellValue(row[col])}
                        </td>
                    `).join('')}
                </tr>
            `;
        });

        html += `
                    </tbody>
                </table>
            </div>
            <div class="data-info">
                <span>Showing ${preview.length} of ${this.currentData.rows} rows</span>
                <button class="btn btn-sm" onclick="openStatica.dataModule.loadMoreData()">
                    Load More
                </button>
            </div>
        `;

        container.innerHTML = html;

        // Add column sorting
        this.setupColumnSorting();
    }

    getColumnType(columnName) {
        if (!this.currentData) return 'unknown';

        if (this.currentData.numeric_columns.includes(columnName)) {
            return 'numeric';
        } else if (this.currentData.categorical_columns.includes(columnName)) {
            return 'categorical';
        } else if (this.currentData.datetime_columns?.includes(columnName)) {
            return 'datetime';
        }
        return 'text';
    }

    getCellClass(value) {
        if (value === null || value === undefined) return 'null-value';
        if (typeof value === 'number') return 'numeric-value';
        if (typeof value === 'boolean') return 'boolean-value';
        return 'text-value';
    }

    formatCellValue(value) {
        if (value === null || value === undefined) return '<span class="null">null</span>';
        if (typeof value === 'number') {
            if (Number.isInteger(value)) return value.toString();
            return value.toFixed(4);
        }
        if (typeof value === 'boolean') return value ? '✓' : '✗';
        if (value.length > 50) return value.substring(0, 50) + '...';
        return value;
    }

    displayDataStatistics(dataInfo) {
        const statsContainer = document.createElement('div');
        statsContainer.className = 'data-statistics';

        statsContainer.innerHTML = `
            <h3>Data Overview</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Rows</div>
                    <div class="stat-value">${dataInfo.rows.toLocaleString()}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Columns</div>
                    <div class="stat-value">${dataInfo.columns}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Numeric Columns</div>
                    <div class="stat-value">${dataInfo.numeric_columns.length}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Categorical Columns</div>
                    <div class="stat-value">${dataInfo.categorical_columns.length}</div>
                </div>
            </div>
            
            ${this.createMissingValuesChart(dataInfo)}
            ${this.createDataTypesChart(dataInfo)}
        `;

        const existingStats = document.querySelector('.data-statistics');
        if (existingStats) {
            existingStats.replaceWith(statsContainer);
        } else {
            document.getElementById('dataView').appendChild(statsContainer);
        }
    }

    createMissingValuesChart(dataInfo) {
        // Create a simple bar chart for missing values
        if (!dataInfo.missing_values) return '';

        let html = '<div class="missing-values-chart"><h4>Missing Values</h4>';

        for (const [column, count] of Object.entries(dataInfo.missing_values)) {
            if (count > 0) {
                const percentage = (count / dataInfo.rows * 100).toFixed(1);
                html += `
                    <div class="missing-bar">
                        <span class="column-name">${column}</span>
                        <div class="bar-container">
                            <div class="bar" style="width: ${percentage}%"></div>
                            <span class="percentage">${percentage}%</span>
                        </div>
                    </div>
                `;
            }
        }

        html += '</div>';
        return html;
    }

    createDataTypesChart(dataInfo) {
        // Create a pie chart for data types distribution
        const types = {
            'Numeric': dataInfo.numeric_columns.length,
            'Categorical': dataInfo.categorical_columns.length,
            'DateTime': dataInfo.datetime_columns?.length || 0
        };

        return `
            <div class="data-types-chart">
                <h4>Column Types Distribution</h4>
                <canvas id="dataTypesChart"></canvas>
            </div>
        `;
    }

    updateVariableList(dataInfo) {
        const container = document.getElementById('variableList');
        if (!container) return;

        container.innerHTML = '';

        // Group variables by type
        const groups = [
            {title: 'Numeric Variables', items: dataInfo.numeric_columns, type: 'numeric'},
            {title: 'Categorical Variables', items: dataInfo.categorical_columns, type: 'categorical'},
            {title: 'DateTime Variables', items: dataInfo.datetime_columns || [], type: 'datetime'}
        ];

        groups.forEach(group => {
            if (group.items.length === 0) return;

            const groupDiv = document.createElement('div');
            groupDiv.className = 'variable-group';

            groupDiv.innerHTML = `
                <div class="variable-group-header">
                    <span>${group.title} (${group.items.length})</span>
                    <button class="btn-text" onclick="openStatica.dataModule.selectAllInGroup('${group.type}')">
                        Select All
                    </button>
                </div>
            `;

            group.items.forEach(variable => {
                const varItem = document.createElement('div');
                varItem.className = 'variable-item';
                varItem.innerHTML = `
                    <input type="checkbox" id="var_${variable}" value="${variable}">
                    <label for="var_${variable}">${variable}</label>
                    <span class="variable-type">${group.type}</span>
                    <button class="btn-icon-sm" onclick="openStatica.dataModule.showVariableInfo('${variable}')" title="Variable Info">
                        ℹ️
                    </button>
                `;
                groupDiv.appendChild(varItem);
            });

            container.appendChild(groupDiv);
        });
    }

    setupColumnSorting() {
        const headers = document.querySelectorAll('.data-table th');
        headers.forEach((header, index) => {
            if (index === 0) return; // Skip row number column

            header.style.cursor = 'pointer';
            header.addEventListener('click', () => {
                this.sortByColumn(index - 1);
            });
        });
    }

    sortByColumn(columnIndex) {
        // TODO: Implement column sorting
        console.log(`Sorting by column ${columnIndex}`);
    }

    async loadMoreData() {
        // TODO: Implement pagination
        console.log('Loading more data...');
    }

    selectAllInGroup(type) {
        let columns = [];
        if (type === 'numeric') {
            columns = this.currentData.numeric_columns;
        } else if (type === 'categorical') {
            columns = this.currentData.categorical_columns;
        } else if (type === 'datetime') {
            columns = this.currentData.datetime_columns || [];
        }

        columns.forEach(col => {
            const checkbox = document.getElementById(`var_${col}`);
            if (checkbox) checkbox.checked = true;
        });
    }

    async showVariableInfo(variable) {
        // Get detailed statistics for the variable
        try {
            const result = await this.app.api.descriptiveStats(
                this.app.sessionId,
                {columns: [variable], options: {include_advanced: true}}
            );

            // Display in a modal or popup
            this.displayVariableModal(variable, result);
        } catch (error) {
            console.error('Failed to get variable info:', error);
        }
    }

    displayVariableModal(variable, stats) {
        // Create modal
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Variable Information: ${variable}</h3>
                    <button class="modal-close" onclick="this.parentElement.parentElement.parentElement.remove()">×</button>
                </div>
                <div class="modal-body">
                    ${this.formatVariableStats(stats)}
                </div>
            </div>
        `;

        document.body.appendChild(modal);
    }

    formatVariableStats(stats) {
        // Format statistics for display
        let html = '<div class="variable-stats">';

        if (stats.results && stats.results[Object.keys(stats.results)[0]]) {
            const varStats = stats.results[Object.keys(stats.results)[0]];

            for (const [key, value] of Object.entries(varStats)) {
                if (typeof value === 'object') continue;

                html += `
                    <div class="stat-row">
                        <span class="stat-label">${this.formatStatName(key)}</span>
                        <span class="stat-value">${this.formatStatValue(value)}</span>
                    </div>
                `;
            }
        }

        html += '</div>';
        return html;
    }

    formatStatName(name) {
        return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    formatStatValue(value) {
        if (typeof value === 'number') {
            return value.toFixed(4);
        }
        return value;
    }

    async transformData(transformationType) {
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Data Transformation</h3>
                    <button class="modal-close" onclick="this.parentElement.parentElement.parentElement.remove()">×</button>
                </div>
                <div class="modal-body">
                    ${this.getTransformationForm(transformationType)}
                </div>
                <div class="modal-footer">
                    <button class="btn btn-primary" onclick="openStatica.dataModule.applyTransformation()">
                        Apply
                    </button>
                    <button class="btn btn-secondary" onclick="this.parentElement.parentElement.parentElement.remove()">
                        Cancel
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
    }

    getTransformationForm(type) {
        const forms = {
            'normalize': `
                <div class="form-group">
                    <label>Normalization Method</label>
                    <select id="normMethod">
                        <option value="zscore">Z-Score</option>
                        <option value="minmax">Min-Max</option>
                        <option value="robust">Robust Scaler</option>
                    </select>
                </div>
            `,
            'encode': `
                <div class="form-group">
                    <label>Encoding Method</label>
                    <select id="encodeMethod">
                        <option value="onehot">One-Hot Encoding</option>
                        <option value="label">Label Encoding</option>
                        <option value="ordinal">Ordinal Encoding</option>
                    </select>
                </div>
            `,
            'impute': `
                <div class="form-group">
                    <label>Imputation Method</label>
                    <select id="imputeMethod">
                        <option value="mean">Mean</option>
                        <option value="median">Median</option>
                        <option value="mode">Mode</option>
                        <option value="forward">Forward Fill</option>
                        <option value="backward">Backward Fill</option>
                        <option value="interpolate">Interpolate</option>
                    </select>
                </div>
            `
        };

        return forms[type] || '<p>Transformation not implemented</p>';
    }

    async applyTransformation() {
        // TODO: Implement transformation application
        console.log('Applying transformation...');
    }
}