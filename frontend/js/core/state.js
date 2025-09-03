/**
 * State Management for OpenStatica
 * Central state management system
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
            theme: localStorage.getItem('theme') || 'light',
            currentView: 'data',
            ready: false
        });

        this.set('session', {
            id: null,
            created: null,
            lastActivity: null
        });

        this.set('data', {
            loaded: false,
            sessionId: null,
            info: null,
            preview: null,
            columns: [],
            rows: 0
        });

        this.set('analysis', {
            current: null,
            history: [],
            results: {},
            selectedVariables: []
        });

        this.set('models', {
            trained: [],
            current: null,
            available: []
        });

        this.set('ui', {
            loading: false,
            sidebarCollapsed: false,
            modalOpen: false,
            notifications: []
        });

        this.set('features', {
            ml: true,
            plugins: true,
            gpu: false
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
        this.addToHistory({
            key,
            oldValue,
            newValue: value,
            timestamp: Date.now()
        });

        // Notify listeners
        if (!silent) {
            this.notify(key, value, oldValue);
        }

        // Persist certain keys
        this.persist(key, value);

        return value;
    }

    update(key, updates, silent = false) {
        const current = this.get(key) || {};
        const updated = {...current, ...updates};
        return this.set(key, updated, silent);
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
        // Notify specific listeners
        const callbacks = this.listeners.get(key) || [];
        callbacks.forEach(callback => {
            try {
                callback(newValue, oldValue, key);
            } catch (error) {
                console.error(`Error in state listener for ${key}:`, error);
            }
        });

        // Notify global listeners
        const globalCallbacks = this.listeners.get('*') || [];
        globalCallbacks.forEach(callback => {
            try {
                callback({key, newValue, oldValue});
            } catch (error) {
                console.error('Error in global state listener:', error);
            }
        });
    }

    addToHistory(change) {
        this.history.push(change);

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
        // Only persist certain keys to localStorage
        const persistKeys = ['app', 'ui'];

        if (persistKeys.includes(key)) {
            try {
                localStorage.setItem(`openstatica_${key}`, JSON.stringify(value));
            } catch (error) {
                console.error(`Failed to persist ${key}:`, error);
            }
        }
    }

    restore() {
        // Restore persisted state from localStorage
        ['app', 'ui'].forEach(key => {
            try {
                const stored = localStorage.getItem(`openstatica_${key}`);
                if (stored) {
                    const value = JSON.parse(stored);
                    this.set(key, value, true);
                }
            } catch (error) {
                console.error(`Failed to restore ${key}:`, error);
            }
        });
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

    // Helper methods
    isDataLoaded() {
        return this.get('data', 'loaded');
    }

    getSessionId() {
        return this.get('session', 'id');
    }

    setLoading(loading) {
        this.update('ui', {loading});
    }

    addNotification(message, type = 'info') {
        const notifications = this.get('ui', 'notifications') || [];
        notifications.push({
            id: Date.now(),
            message,
            type,
            timestamp: new Date()
        });
        this.update('ui', {notifications});
    }

    clearNotifications() {
        this.update('ui', {notifications: []});
    }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = StateManager;
}