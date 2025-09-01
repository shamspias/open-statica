/**
 * State Management for OpenStatica
 */

class StateManager {
    constructor() {
        this.state = new Map();
        this.listeners = new Map();
        this.history = [];
        this.maxHistory = 50;

        // Initialize default state
        this.initializeState();
    }

    initializeState() {
        this.set('app', {
            version: '1.0.0',
            theme: 'light',
            currentView: 'data'
        });

        this.set('data', {
            loaded: false,
            sessionId: null,
            info: null,
            preview: null
        });

        this.set('analysis', {
            lastRun: null,
            results: {},
            selectedVariables: []
        });

        this.set('models', {
            trained: [],
            current: null
        });

        this.set('ui', {
            loading: false,
            sidebarCollapsed: false,
            modalOpen: false
        });
    }

    get(key, path = null) {
        const value = this.state.get(key);

        if (path && value) {
            return this.getNestedValue(value, path);
        }

        return value;
    }

    set(key, value, silent = false) {
        const oldValue = this.state.get(key);
        this.state.set(key, value);

        // Add to history
        this.addToHistory({key, oldValue, newValue: value});

        // Notify listeners
        if (!silent) {
            this.notify(key, value, oldValue);
        }

        // Persist to localStorage if needed
        this.persist(key, value);
    }

    update(key, updates, silent = false) {
        const current = this.get(key) || {};
        const updated = {...current, ...updates};
        this.set(key, updated, silent);
    }

    getNestedValue(obj, path) {
        const keys = path.split('.');
        let value = obj;

        for (const key of keys) {
            value = value?.[key];
            if (value === undefined) break;
        }

        return value;
    }

    setNestedValue(key, path, value) {
        const current = this.get(key) || {};
        const keys = path.split('.');
        const lastKey = keys.pop();

        let target = current;
        for (const k of keys) {
            if (!target[k]) target[k] = {};
            target = target[k];
        }

        target[lastKey] = value;
        this.set(key, current);
    }

    subscribe(key, callback) {
        if (!this.listeners.has(key)) {
            this.listeners.set(key, []);
        }

        this.listeners.get(key).push(callback);

        // Return unsubscribe function
        return () => {
            const callbacks = this.listeners.get(key);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        };
    }

    notify(key, newValue, oldValue) {
        const callbacks = this.listeners.get(key) || [];
        callbacks.forEach(callback => {
            callback(newValue, oldValue, key);
        });

        // Notify global listeners
        const globalCallbacks = this.listeners.get('*') || [];
        globalCallbacks.forEach(callback => {
            callback({key, newValue, oldValue});
        });
    }

    addToHistory(change) {
        this.history.push({
            ...change,
            timestamp: Date.now()
        });

        // Limit history size
        if (this.history.length > this.maxHistory) {
            this.history.shift();
        }
    }

    undo() {
        if (this.history.length === 0) return false;

        const lastChange = this.history.pop();
        this.set(lastChange.key, lastChange.oldValue, true);
        return true;
    }

    persist(key, value) {
        // Only persist certain keys
        const persistKeys = ['app', 'ui'];

        if (persistKeys.includes(key)) {
            utils.storage.set(`openstatica_${key}`, value);
        }
    }

    restore() {
        // Restore persisted state
        const appState = utils.storage.get('openstatica_app');
        if (appState) {
            this.set('app', appState, true);
        }

        const uiState = utils.storage.get('openstatica_ui');
        if (uiState) {
            this.set('ui', uiState, true);
        }
    }

    reset() {
        this.state.clear();
        this.history = [];
        this.initializeState();
    }

    export() {
        const exportData = {};
        this.state.forEach((value, key) => {
            exportData[key] = value;
        });
        return exportData;
    }

    import(data) {
        Object.entries(data).forEach(([key, value]) => {
            this.set(key, value, true);
        });
    }
}

// Create global state manager instance
window.stateManager = new StateManager();