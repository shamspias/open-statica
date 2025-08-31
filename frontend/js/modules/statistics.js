/**
 * Statistics Module for OpenStatica
 */

class StatisticsModule {
    constructor(app) {
        this.app = app;
        this.currentTest = null;
    }

    loadDescriptiveOptions() {
        const container = document.getElementById('optionsContent');
        if (!container) return;

        container.innerHTML = `
            <div class="form-group">
                <label class="form-label">Analysis Type</label>
                <select class="form-control" id="descriptiveType">
                    <option value="basic">Basic Statistics</option>
                    <option value="frequency">Frequency Distribution</option>
                    <option value="crosstab">Cross-tabulation</option>
                    <option value="normality">Normality Tests</option>
                    <option value="outliers">Outlier Detection</option>
                </select>
            </div>
            
            <div class="form-group">
                <label class="form-checkbox">
                    <input type="checkbox" id="includeAdvanced" checked>
                    <span>Include advanced statistics</span>
                </label>
            </div>
            
            <div class="form-group">
                <label class="form-checkbox">
                    <input type="checkbox" id="includeVisualizations" checked>
                    <span>Generate visualizations</span>
                </label>
            </div>
            
            <div class="form-group">
                <label class="form-label">Confidence Level</label>
                <select class="form-control" id="confidenceLevel">
                    <option value="0.90">90%</option>
                    <option value="0.95" selected>95%</option>
                    <option value="0.99">99%</option>
                </select>
            </div>
        `;

        this.bindDescriptiveEvents();
    }

    loadInferentialOptions() {
        const container = document.getElementById('optionsContent');
        if (!container) return;

        container.innerHTML = `
            <div class="form-group">
                <label class="form-label">Test Type</label>
                <select class="form-control" id="inferentialTest">
                    <optgroup label="Parametric Tests">
                        <option value="ttest_one">One-Sample T-Test</option>
                        <option value="ttest_independent">Independent T-Test</option>
                        <option value="ttest_paired">Paired T-Test</option>
                        <option value="anova_one">One-Way ANOVA</option>
                        <option value="anova_two">Two-Way ANOVA</option>
                        <option value="anova_repeated">Repeated Measures ANOVA</option>
                    </optgroup>
                    <optgroup label="Non-Parametric Tests">
                        <option value="mann_whitney">Mann-Whitney U</option>
                        <option value="wilcoxon">Wilcoxon Signed-Rank</option>
                        <option value="kruskal">Kruskal-Wallis</option>
                        <option value="friedman">Friedman Test</option>
                    </optgroup>
                    <optgroup label="Correlation Tests">
                        <option value="pearson">Pearson Correlation</option>
                        <option value="spearman">Spearman Correlation</option>
                        <option value="kendall">Kendall's Tau</option>
                    </optgroup>
                    <optgroup label="Other Tests">
                        <option value="chi_square">Chi-Square Test</option>
                        <option value="fisher">Fisher's Exact Test</option>
                        <option value="mcnemar">McNemar's Test</option>
                    </optgroup>
                </select>
            </div>
            
            <div id="testSpecificOptions"></div>
            
            <div class="form-group">
                <label class="form-label">Significance Level (α)</label>
                <select class="form-control" id="alphaLevel">
                    <option value="0.01">0.01</option>
                    <option value="0.05" selected>0.05</option>
                    <option value="0.10">0.10</option>
                </select>
            </div>
            
            <div class="form-group">
                <label class="form-checkbox">
                    <input type="checkbox" id="includePostHoc" checked>
                    <span>Include post-hoc tests (if applicable)</span>
                </label>
            </div>
            
            <div class="form-group">
                <label class="form-checkbox">
                    <input type="checkbox" id="includeEffectSize" checked>
                    <span>Calculate effect sizes</span>
                </label>
            </div>
        `;

        this.bindInferentialEvents();
    }

    loadRegressionOptions() {
        const container = document.getElementById('optionsContent');
        if (!container) return;

        container.innerHTML = `
            <div class="form-group">
                <label class="form-label">Regression Type</label>
                <select class="form-control" id="regressionType">
                    <optgroup label="Linear Models">
                        <option value="linear">Linear Regression</option>
                        <option value="multiple">Multiple Linear Regression</option>
                        <option value="polynomial">Polynomial Regression</option>
                        <option value="stepwise">Stepwise Regression</option>
                    </optgroup>
                    <optgroup label="Regularized Models">
                        <option value="ridge">Ridge Regression</option>
                        <option value="lasso">Lasso Regression</option>
                        <option value="elastic">Elastic Net</option>
                    </optgroup>
                    <optgroup label="Generalized Linear Models">
                        <option value="logistic">Logistic Regression</option>
                        <option value="poisson">Poisson Regression</option>
                        <option value="gamma">Gamma Regression</option>
                    </optgroup>
                    <optgroup label="Advanced Models">
                        <option value="quantile">Quantile Regression</option>
                        <option value="robust">Robust Regression</option>
                        <option value="pls">Partial Least Squares</option>
                    </optgroup>
                </select>
            </div>
            
            <div class="form-group">
                <label class="form-label">Dependent Variable</label>
                <select class="form-control" id="dependentVariable">
                    <option value="">Select variable...</option>
                    ${this.getNumericVariableOptions()}
                </select>
            </div>
            
            <div class="form-group">
                <label class="form-label">Feature Selection</label>
                <select class="form-control" id="featureSelection">
                    <option value="none">None</option>
                    <option value="forward">Forward Selection</option>
                    <option value="backward">Backward Elimination</option>
                    <option value="both">Bidirectional</option>
                    <option value="rfe">Recursive Feature Elimination</option>
                </select>
            </div>
            
            <div class="form-group">
                <label class="form-checkbox">
                    <input type="checkbox" id="includeDiagnostics" checked>
                    <span>Include diagnostic plots</span>
                </label>
            </div>
            
            <div class="form-group">
                <label class="form-checkbox">
                    <input type="checkbox" id="crossValidation" checked>
                    <span>Perform cross-validation</span>
                </label>
            </div>
        `;

        this.bindRegressionEvents();
    }

    bindDescriptiveEvents() {
        document.getElementById('descriptiveType')?.addEventListener('change', (e) => {
            this.updateDescriptiveSubOptions(e.target.value);
        });
    }

    bindInferentialEvents() {
        document.getElementById('inferentialTest')?.addEventListener('change', (e) => {
            this.updateTestSpecificOptions(e.target.value);
        });
    }

    bindRegressionEvents() {
        document.getElementById('regressionType')?.addEventListener('change', (e) => {
            this.updateRegressionSubOptions(e.target.value);
        });
    }

    updateTestSpecificOptions(testType) {
        const container = document.getElementById('testSpecificOptions');
        if (!container) return;

        let html = '';

        if (testType.startsWith('ttest_one')) {
            html = `
                <div class="form-group">
                    <label class="form-label">Test Value</label>
                    <input type="number" class="form-control" id="testValue" value="0">
                </div>
            `;
        } else if (testType.includes('anova')) {
            html = `
                <div class="form-group">
                    <label class="form-label">Post-hoc Test</label>
                    <select class="form-control" id="postHocTest">
                        <option value="tukey">Tukey HSD</option>
                        <option value="bonferroni">Bonferroni</option>
                        <option value="scheffe">Scheffe</option>
                        <option value="sidak">Sidak</option>
                    </select>
                </div>
            `;
        }

        container.innerHTML = html;
    }

    getNumericVariableOptions() {
        if (!this.app.data) return '';

        return this.app.data.numeric_columns.map(col =>
            `<option value="${col}">${col}</option>`
        ).join('');
    }

    async runDescriptiveAnalysis(variables) {
        const type = document.getElementById('descriptiveType')?.value || 'basic';
        const includeAdvanced = document.getElementById('includeAdvanced')?.checked;
        const includeViz = document.getElementById('includeVisualizations')?.checked;

        const params = {
            columns: variables,
            options: {
                type: type,
                include_advanced: includeAdvanced,
                include_visualizations: includeViz
            }
        };

        return await this.app.api.descriptiveStats(this.app.sessionId, params);
    }

    async runInferentialAnalysis(variables) {
        const testType = document.getElementById('inferentialTest')?.value;
        const alpha = parseFloat(document.getElementById('alphaLevel')?.value || 0.05);

        const params = {
            columns: variables,
            options: {
                test_type: testType,
                alpha: alpha,
                include_post_hoc: document.getElementById('includePostHoc')?.checked,
                include_effect_size: document.getElementById('includeEffectSize')?.checked
            }
        };

        // Add test-specific parameters
        if (testType === 'ttest_one') {
            params.options.test_value = parseFloat(document.getElementById('testValue')?.value || 0);
        }

        return await this.app.api.tTest(this.app.sessionId, params);
    }

    async runRegressionAnalysis(variables) {
        const regressionType = document.getElementById('regressionType')?.value;
        const dependent = document.getElementById('dependentVariable')?.value;

        if (!dependent) {
            throw new Error('Please select a dependent variable');
        }

        const params = {
            columns: variables,
            options: {
                method: regressionType,
                dependent: dependent,
                independent: variables.filter(v => v !== dependent),
                feature_selection: document.getElementById('featureSelection')?.value,
                include_diagnostics: document.getElementById('includeDiagnostics')?.checked,
                cross_validation: document.getElementById('crossValidation')?.checked
            }
        };

        return await this.app.api.regression(this.app.sessionId, params);
    }

    formatStatisticalResults(results) {
        // Create formatted HTML for results display
        let html = '<div class="statistical-results">';

        // Add summary
        if (results.summary) {
            html += `
                <div class="result-summary">
                    <h3>Summary</h3>
                    <p>${results.summary}</p>
                </div>
            `;
        }

        // Add main statistics
        if (results.statistics) {
            html += '<div class="result-statistics">';
            html += this.formatStatisticsTable(results.statistics);
            html += '</div>';
        }

        // Add interpretation
        if (results.interpretation) {
            html += `
                <div class="result-interpretation">
                    <h3>Interpretation</h3>
                    <p>${results.interpretation}</p>
                </div>
            `;
        }

        html += '</div>';
        return html;
    }

    formatStatisticsTable(stats) {
        let html = '<table class="statistics-table">';
        html += '<thead><tr><th>Statistic</th><th>Value</th></tr></thead>';
        html += '<tbody>';

        for (const [key, value] of Object.entries(stats)) {
            const formattedKey = this.formatStatisticName(key);
            const formattedValue = this.formatStatisticValue(value);
            html += `<tr><td>${formattedKey}</td><td>${formattedValue}</td></tr>`;
        }

        html += '</tbody></table>';
        return html;
    }

    formatStatisticName(name) {
        return name
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
    }

    formatStatisticValue(value) {
        if (typeof value === 'number') {
            return value.toFixed(4);
        } else if (Array.isArray(value)) {
            return `[${value.map(v => v.toFixed(4)).join(', ')}]`;
        } else if (typeof value === 'object') {
            return JSON.stringify(value, null, 2);
        }
        return value;
    }
}