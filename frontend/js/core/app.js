/**
 * OpenStatica - Main Application Class
 * Open-source statistical & ML platform
 */

class OpenStatica {
    constructor() {
        this.version = "1.0.0";
        this.sessionId = null;
        this.currentView = 'data';
        this.data = null;

        // Initialize core modules
        this.api = new OpenStaticaAPI();
        this.state = new StateManager();
        this.statistics = new StatisticsModule(this);
        this.ml = new MLModule(this);
        this.visualization = new VisualizationModule(this);
        this.dataModule = new DataModule(this);

        // Plugin system
        this.plugins = new Map();

        this.initialize();
    }

    async initialize() {
        console.log(`🚀 OpenStatica v${this.version} initializing...`);

        // Check backend connection
        await this.checkBackendStatus();

        // Initialize UI
        this.initializeUI();
        this.bindEventListeners();

        // Load plugins if available
        await this.loadPlugins();

        console.log('✅ OpenStatica ready');
    }

    async checkBackendStatus() {
        try {
            const status = await this.api.getStatus();
            console.log('Backend status:', status);

            // Update UI based on available features
            if (status.features) {
                this.updateFeatureAvailability(status.features);
            }
        } catch (error) {
            console.error('Failed to connect to backend:', error);
            this.showNotification('Failed to connect to backend', 'error');
        }
    }

    initializeUI() {
        // Set up initial UI state
        this.updateViewVisibility();
        this.setupTheme();
        this.setupTooltips();
    }

    bindEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchView(e.target.dataset.view);
            });
        });

        // File upload
        this.setupFileUpload();

        // Settings
        document.getElementById('settingsBtn')?.addEventListener('click', () => {
            this.openSettings();
        });

        // Global keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcut(e);
        });
    }

    setupFileUpload() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        if (!uploadArea || !fileInput) return;

        // Click to upload
        uploadArea.addEventListener('click', () => fileInput.click());

        // File selection
        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) await this.handleFileUpload(file);
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', async (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');

            const file = e.dataTransfer.files[0];
            if (file) await this.handleFileUpload(file);
        });
    }

    async handleFileUpload(file) {
        try {
            this.showLoading(true);

            // Validate file
            if (!this.validateFile(file)) {
                throw new Error('Invalid file type or size');
            }

            // Upload file
            const response = await this.api.uploadFile(file);

            // Store session info
            this.sessionId = response.session_id;
            this.data = response;
            this.state.set('currentSession', response);

            // Update UI
            this.dataModule.displayData(response);
            this.updateUIAfterDataLoad();

            this.showNotification(`Successfully loaded ${response.rows} rows × ${response.columns} columns`, 'success');

        } catch (error) {
            console.error('File upload error:', error);
            this.showNotification(error.message, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    validateFile(file) {
        const maxSize = 100 * 1024 * 1024; // 100MB
        const allowedTypes = ['.csv', '.xlsx', '.xls', '.json', '.parquet'];

        const extension = '.' + file.name.split('.').pop().toLowerCase();

        if (!allowedTypes.includes(extension)) {
            this.showNotification(`File type ${extension} not supported`, 'error');
            return false;
        }

        if (file.size > maxSize) {
            this.showNotification('File size exceeds 100MB limit', 'error');
            return false;
        }

        return true;
    }

    updateUIAfterDataLoad() {
        // Show data info
        document.getElementById('dataInfo').textContent =
            `${this.data.rows} rows × ${this.data.columns} columns`;

        // Show file info
        document.getElementById('fileInfo').style.display = 'block';

        // Enable analysis options
        document.getElementById('variableSelector').style.display = 'block';

        // Populate variables
        this.populateVariables();

        // Enable relevant navigation
        this.enableNavigation();
    }

    populateVariables() {
        const container = document.getElementById('variableList');
        if (!container) return;

        container.innerHTML = '';

        // Add numeric variables
        this.data.numeric_columns.forEach(col => {
            container.appendChild(this.createVariableElement(col, 'numeric'));
        });

        // Add categorical variables
        this.data.categorical_columns.forEach(col => {
            container.appendChild(this.createVariableElement(col, 'categorical'));
        });

        // Add datetime variables if any
        if (this.data.datetime_columns) {
            this.data.datetime_columns.forEach(col => {
                container.appendChild(this.createVariableElement(col, 'datetime'));
            });
        }
    }

    createVariableElement(name, type) {
        const div = document.createElement('div');
        div.className = 'variable-item';
        div.innerHTML = `
            <input type="checkbox" id="var_${name}" value="${name}">
            <label for="var_${name}">${name}</label>
            <span class="variable-type">${type}</span>
        `;
        return div;
    }

    switchView(view) {
        if (!view) return;

        this.currentView = view;

        // Update navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === view);
        });

        // Update view panels
        document.querySelectorAll('.view-panel').forEach(panel => {
            panel.classList.toggle('active', panel.id === `${view}View`);
        });

        // Load view-specific options
        this.loadViewOptions(view);

        // Track view change
        this.state.set('currentView', view);
    }

    loadViewOptions(view) {
        const optionsContainer = document.getElementById('analysisOptions');
        if (!optionsContainer) return;

        // Show/hide options based on view
        optionsContainer.style.display =
            ['descriptive', 'inferential', 'regression', 'multivariate', 'ml'].includes(view)
                ? 'block' : 'none';

        // Load specific options for each view
        switch (view) {
            case 'descriptive':
                this.statistics.loadDescriptiveOptions();
                break;
            case 'inferential':
                this.statistics.loadInferentialOptions();
                break;
            case 'regression':
                this.statistics.loadRegressionOptions();
                break;
            case 'ml':
                this.ml.loadMLOptions();
                break;
        }
    }

    updateViewVisibility() {
        // Show only active view
        const activeView = this.state.get('currentView') || 'data';
        this.switchView(activeView);
    }

    setupTheme() {
        const theme = localStorage.getItem('theme') || 'light';
        document.body.dataset.theme = theme;
    }

    setupTooltips() {
        // Initialize tooltips for help icons
        document.querySelectorAll('[data-tooltip]').forEach(el => {
            el.title = el.dataset.tooltip;
        });
    }

    enableNavigation() {
        // Enable navigation buttons based on data availability
        document.querySelectorAll('.nav-btn').forEach(btn => {
            if (btn.dataset.view !== 'data') {
                btn.disabled = false;
            }
        });
    }

    async loadPlugins() {
        try {
            // Check for available plugins
            const plugins = await this.api.getAvailablePlugins();

            if (plugins && plugins.length > 0) {
                console.log(`Loading ${plugins.length} plugins...`);

                for (const plugin of plugins) {
                    await this.loadPlugin(plugin);
                }
            }
        } catch (error) {
            console.log('No plugins available or plugin system disabled');
        }
    }

    async loadPlugin(pluginInfo) {
        try {
            // Dynamic import of plugin module
            const module = await import(`/plugins/${pluginInfo.name}/index.js`);
            const Plugin = module.default;

            const plugin = new Plugin(this);
            await plugin.initialize();

            this.plugins.set(pluginInfo.name, plugin);
            console.log(`✓ Loaded plugin: ${pluginInfo.name}`);

        } catch (error) {
            console.error(`Failed to load plugin ${pluginInfo.name}:`, error);
        }
    }

    handleKeyboardShortcut(e) {
        // Ctrl/Cmd + O: Open file
        if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
            e.preventDefault();
            document.getElementById('fileInput')?.click();
        }

        // Ctrl/Cmd + S: Save results
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            this.saveResults();
        }

        // Escape: Close modals
        if (e.key === 'Escape') {
            this.closeAllModals();
        }
    }

    openSettings() {
        // TODO: Implement settings modal
        console.log('Settings not yet implemented');
    }

    async saveResults() {
        // TODO: Implement results export
        console.log('Save results not yet implemented');
    }

    closeAllModals() {
        document.querySelectorAll('.modal').forEach(modal => {
            modal.style.display = 'none';
        });
    }

    updateFeatureAvailability(features) {
        // Update UI based on available backend features
        if (!features.ml) {
            document.querySelector('[data-view="ml"]')?.setAttribute('disabled', 'true');
        }

        this.state.set('features', features);
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;

        // Add to notification container or create one
        let container = document.getElementById('notifications');
        if (!container) {
            container = document.createElement('div');
            container.id = 'notifications';
            document.body.appendChild(container);
        }

        container.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }

    showLoading(show = true) {
        let loader = document.getElementById('globalLoader');

        if (show) {
            if (!loader) {
                loader = document.createElement('div');
                loader.id = 'globalLoader';
                loader.className = 'loading-overlay';
                loader.innerHTML = '<div class="loading-spinner"></div>';
                document.body.appendChild(loader);
            }
            loader.style.display = 'flex';
        } else {
            if (loader) {
                loader.style.display = 'none';
            }
        }
    }

    getSelectedVariables() {
        const checkboxes = document.querySelectorAll('#variableList input:checked');
        return Array.from(checkboxes).map(cb => cb.value);
    }

    async runAnalysis() {
        const variables = this.getSelectedVariables();

        if (variables.length === 0) {
            this.showNotification('Please select at least one variable', 'warning');
            return;
        }

        try {
            this.showLoading(true);

            let result;

            switch (this.currentView) {
                case 'descriptive':
                    result = await this.statistics.runDescriptiveAnalysis(variables);
                    break;
                case 'inferential':
                    result = await this.statistics.runInferentialAnalysis(variables);
                    break;
                case 'regression':
                    result = await this.statistics.runRegressionAnalysis(variables);
                    break;
                case 'ml':
                    result = await this.ml.runMLAnalysis(variables);
                    break;
                default:
                    throw new Error('Invalid analysis type');
            }

            // Display results
            this.displayResults(result);

            // Generate visualizations
            await this.visualization.createVisualizations(result);

            this.showNotification('Analysis completed successfully', 'success');

        } catch (error) {
            console.error('Analysis error:', error);
            this.showNotification(error.message, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    displayResults(results) {
        const container = document.getElementById(`${this.currentView}Results`);
        if (!container) return;

        // Clear previous results
        container.innerHTML = '';

        // Create result cards
        const card = document.createElement('div');
        card.className = 'result-card';

        // Format and display results based on type
        card.innerHTML = this.formatResults(results);

        container.appendChild(card);

        // Store results in state
        this.state.set('lastResults', results);
    }

    formatResults(results) {
        // TODO: Implement comprehensive result formatting
        return `<pre>${JSON.stringify(results, null, 2)}</pre>`;
    }
}

// State Manager
class StateManager {
    constructor() {
        this.state = new Map();
    }

    get(key) {
        return this.state.get(key);
    }

    set(key, value) {
        this.state.set(key, value);
        this.emit('stateChange', {key, value});
    }

    emit(event, data) {
        window.dispatchEvent(new CustomEvent(event, {detail: data}));
    }
}

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.openStatica = new OpenStatica();
});