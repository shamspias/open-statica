/**
 * OpenStatica API Client
 * Handles all communication with backend
 */

class OpenStaticaAPI {
    constructor() {
        this.baseURL = process.env.API_URL || 'http://localhost:8000';
        this.apiVersion = 'v1';
        this.timeout = 30000;
        this.headers = {
            'Content-Type': 'application/json',
        };
    }

    /**
     * Generic request handler with error handling
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}/api/${this.apiVersion}${endpoint}`;

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        try {
            const response = await fetch(url, {
                ...options,
                headers: {
                    ...this.headers,
                    ...options.headers,
                },
                signal: controller.signal,
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || error.message || `HTTP ${response.status}`);
            }

            return await response.json();

        } catch (error) {
            clearTimeout(timeoutId);

            if (error.name === 'AbortError') {
                throw new Error('Request timeout');
            }

            throw error;
        }
    }

    // System endpoints
    async getStatus() {
        return await this.request('/', {method: 'GET'});
    }

    async getHealth() {
        return await this.request('/health', {method: 'GET'});
    }

    // Data endpoints
    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        return await fetch(`${this.baseURL}/api/${this.apiVersion}/data/upload`, {
            method: 'POST',
            body: formData,
        }).then(res => res.json());
    }

    async getData(sessionId) {
        return await this.request(`/data/${sessionId}`);
    }

    async deleteSession(sessionId) {
        return await this.request(`/data/${sessionId}`, {method: 'DELETE'});
    }

    async getDataInfo(sessionId) {
        return await this.request(`/data/${sessionId}/info`);
    }

    // Statistical endpoints
    async descriptiveStats(sessionId, params) {
        return await this.request('/statistics/descriptive', {
            method: 'POST',
            body: JSON.stringify({
                session_id: sessionId,
                ...params
            })
        });
    }

    async frequencyDistribution(sessionId, params) {
        return await this.request('/statistics/frequency', {
            method: 'POST',
            body: JSON.stringify({
                session_id: sessionId,
                ...params
            })
        });
    }

    async tTest(sessionId, params) {
        return await this.request('/statistics/ttest', {
            method: 'POST',
            body: JSON.stringify({
                session_id: sessionId,
                ...params
            })
        });
    }

    async anova(sessionId, params) {
        return await this.request('/statistics/anova', {
            method: 'POST',
            body: JSON.stringify({
                session_id: sessionId,
                ...params
            })
        });
    }

    async correlation(sessionId, params) {
        return await this.request('/statistics/correlation', {
            method: 'POST',
            body: JSON.stringify({
                session_id: sessionId,
                ...params
            })
        });
    }

    async regression(sessionId, params) {
        return await this.request('/statistics/regression', {
            method: 'POST',
            body: JSON.stringify({
                session_id: sessionId,
                ...params
            })
        });
    }

    // ML endpoints
    async trainModel(sessionId, params) {
        return await this.request('/ml/train', {
            method: 'POST',
            body: JSON.stringify({
                session_id: sessionId,
                ...params
            })
        });
    }

    async predict(sessionId, modelId, data) {
        return await this.request('/ml/predict', {
            method: 'POST',
            body: JSON.stringify({
                session_id: sessionId,
                model_id: modelId,
                data: data
            })
        });
    }

    async evaluateModel(sessionId, modelId, testData) {
        return await this.request('/ml/evaluate', {
            method: 'POST',
            body: JSON.stringify({
                session_id: sessionId,
                model_id: modelId,
                test_data: testData
            })
        });
    }

    async getAvailableModels() {
        return await this.request('/ml/models');
    }

    async getModelInfo(modelId) {
        return await this.request(`/ml/models/${modelId}`);
    }

    // Model Hub endpoints
    async searchModels(query, source = 'huggingface') {
        return await this.request(`/models/search?q=${query}&source=${source}`);
    }

    async loadModel(modelId, source = 'huggingface') {
        return await this.request('/models/load', {
            method: 'POST',
            body: JSON.stringify({
                model_id: modelId,
                source: source
            })
        });
    }

    // Visualization endpoints
    async generatePlot(sessionId, params) {
        return await this.request('/visualization/plot', {
            method: 'POST',
            body: JSON.stringify({
                session_id: sessionId,
                ...params
            })
        });
    }

    // Export endpoints
    async exportResults(sessionId, format = 'csv') {
        const response = await fetch(
            `${this.baseURL}/api/${this.apiVersion}/export/${sessionId}?format=${format}`
        );
        return await response.blob();
    }

    // Plugin endpoints
    async getAvailablePlugins() {
        return await this.request('/plugins');
    }

    async enablePlugin(pluginName) {
        return await this.request(`/plugins/${pluginName}/enable`, {
            method: 'POST'
        });
    }

    async disablePlugin(pluginName) {
        return await this.request(`/plugins/${pluginName}/disable`, {
            method: 'POST'
        });
    }
}