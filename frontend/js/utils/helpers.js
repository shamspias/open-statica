/**
 * Helper utilities for OpenStatica
 */

const utils = {
    // Storage utilities
    storage: {
        set(key, value) {
            try {
                localStorage.setItem(key, JSON.stringify(value));
                return true;
            } catch (e) {
                console.error('Storage error:', e);
                return false;
            }
        },

        get(key) {
            try {
                const item = localStorage.getItem(key);
                return item ? JSON.parse(item) : null;
            } catch (e) {
                console.error('Storage error:', e);
                return null;
            }
        },

        remove(key) {
            localStorage.removeItem(key);
        },

        clear() {
            localStorage.clear();
        }
    },

    // Format utilities
    format: {
        number(value, decimals = 2) {
            if (value === null || value === undefined) return 'N/A';
            if (typeof value !== 'number') return value;

            if (Math.abs(value) < 0.01 && value !== 0) {
                return value.toExponential(decimals);
            }
            return value.toFixed(decimals);
        },

        percentage(value, decimals = 1) {
            if (typeof value !== 'number') return 'N/A';
            return `${(value * 100).toFixed(decimals)}%`;
        },

        fileSize(bytes) {
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            if (bytes === 0) return '0 Bytes';
            const i = Math.floor(Math.log(bytes) / Math.log(1024));
            return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
        },

        date(date) {
            if (!date) return '';
            const d = new Date(date);
            return d.toLocaleDateString() + ' ' + d.toLocaleTimeString();
        },

        duration(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);

            if (h > 0) return `${h}h ${m}m ${s}s`;
            if (m > 0) return `${m}m ${s}s`;
            return `${s}s`;
        }
    },

    // DOM utilities
    dom: {
        createElement(tag, className, innerHTML) {
            const element = document.createElement(tag);
            if (className) element.className = className;
            if (innerHTML) element.innerHTML = innerHTML;
            return element;
        },

        show(element) {
            if (typeof element === 'string') {
                element = document.getElementById(element);
            }
            if (element) element.style.display = 'block';
        },

        hide(element) {
            if (typeof element === 'string') {
                element = document.getElementById(element);
            }
            if (element) element.style.display = 'none';
        },

        toggle(element) {
            if (typeof element === 'string') {
                element = document.getElementById(element);
            }
            if (element) {
                element.style.display = element.style.display === 'none' ? 'block' : 'none';
            }
        },

        addClass(element, className) {
            if (typeof element === 'string') {
                element = document.getElementById(element);
            }
            if (element) element.classList.add(className);
        },

        removeClass(element, className) {
            if (typeof element === 'string') {
                element = document.getElementById(element);
            }
            if (element) element.classList.remove(className);
        },

        toggleClass(element, className) {
            if (typeof element === 'string') {
                element = document.getElementById(element);
            }
            if (element) element.classList.toggle(className);
        }
    },

    // Data utilities
    data: {
        deepCopy(obj) {
            return JSON.parse(JSON.stringify(obj));
        },

        merge(...objects) {
            return Object.assign({}, ...objects);
        },

        groupBy(array, key) {
            return array.reduce((result, item) => {
                const group = item[key];
                if (!result[group]) result[group] = [];
                result[group].push(item);
                return result;
            }, {});
        },

        sortBy(array, key, order = 'asc') {
            return array.sort((a, b) => {
                if (order === 'asc') {
                    return a[key] > b[key] ? 1 : -1;
                } else {
                    return a[key] < b[key] ? 1 : -1;
                }
            });
        },

        unique(array) {
            return [...new Set(array)];
        },

        flatten(array) {
            return array.reduce((flat, item) => {
                return flat.concat(Array.isArray(item) ? this.flatten(item) : item);
            }, []);
        }
    },

    // Validation utilities
    validate: {
        email(email) {
            const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            return re.test(email);
        },

        number(value) {
            return !isNaN(value) && isFinite(value);
        },

        required(value) {
            return value !== null && value !== undefined && value !== '';
        },

        minLength(value, min) {
            return value && value.length >= min;
        },

        maxLength(value, max) {
            return value && value.length <= max;
        },

        range(value, min, max) {
            return value >= min && value <= max;
        }
    },

    // Async utilities
    async: {
        delay(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        },

        debounce(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        },

        throttle(func, limit) {
            let inThrottle;
            return function (...args) {
                if (!inThrottle) {
                    func.apply(this, args);
                    inThrottle = true;
                    setTimeout(() => inThrottle = false, limit);
                }
            };
        },

        retry(func, times = 3, delay = 1000) {
            return new Promise((resolve, reject) => {
                const attempt = async (n) => {
                    try {
                        const result = await func();
                        resolve(result);
                    } catch (error) {
                        if (n === 1) {
                            reject(error);
                        } else {
                            await this.delay(delay);
                            attempt(n - 1);
                        }
                    }
                };
                attempt(times);
            });
        }
    },

    // Export utilities
    export: {
        toCSV(data, headers) {
            if (!data || data.length === 0) return '';

            headers = headers || Object.keys(data[0]);

            const csvHeaders = headers.join(',');
            const csvRows = data.map(row =>
                headers.map(header => {
                    const value = row[header];
                    return typeof value === 'string' && value.includes(',')
                        ? `"${value}"`
                        : value;
                }).join(',')
            );

            return [csvHeaders, ...csvRows].join('\n');
        },

        toJSON(data, pretty = true) {
            return pretty ? JSON.stringify(data, null, 2) : JSON.stringify(data);
        },

        download(content, filename, type = 'text/plain') {
            const blob = new Blob([content], {type});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    },

    // Statistics utilities
    stats: {
        mean(array) {
            if (!array || array.length === 0) return null;
            return array.reduce((sum, val) => sum + val, 0) / array.length;
        },

        median(array) {
            if (!array || array.length === 0) return null;
            const sorted = [...array].sort((a, b) => a - b);
            const mid = Math.floor(sorted.length / 2);
            return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
        },

        mode(array) {
            if (!array || array.length === 0) return null;
            const frequency = {};
            let maxFreq = 0;
            let mode = null;

            array.forEach(val => {
                frequency[val] = (frequency[val] || 0) + 1;
                if (frequency[val] > maxFreq) {
                    maxFreq = frequency[val];
                    mode = val;
                }
            });

            return mode;
        },

        standardDeviation(array) {
            if (!array || array.length === 0) return null;
            const avg = this.mean(array);
            const squareDiffs = array.map(val => Math.pow(val - avg, 2));
            return Math.sqrt(this.mean(squareDiffs));
        },

        percentile(array, p) {
            if (!array || array.length === 0) return null;
            const sorted = [...array].sort((a, b) => a - b);
            const index = (p / 100) * (sorted.length - 1);
            const lower = Math.floor(index);
            const upper = Math.ceil(index);
            const weight = index % 1;

            if (lower === upper) return sorted[lower];
            return sorted[lower] * (1 - weight) + sorted[upper] * weight;
        }
    },

    // Color utilities
    colors: {
        palette: [
            '#6366f1', '#8b5cf6', '#ec4899', '#f43f5e',
            '#f97316', '#eab308', '#84cc16', '#22c55e',
            '#10b981', '#14b8a6', '#06b6d4', '#0ea5e9',
            '#3b82f6', '#6366f1', '#8b5cf6', '#a855f7'
        ],

        getColor(index) {
            return this.palette[index % this.palette.length];
        },

        hexToRgb(hex) {
            const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
            return result ? {
                r: parseInt(result[1], 16),
                g: parseInt(result[2], 16),
                b: parseInt(result[3], 16)
            } : null;
        },

        rgbToHex(r, g, b) {
            return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
        }
    },

    // Event utilities
    events: {
        on(element, event, handler) {
            if (typeof element === 'string') {
                element = document.getElementById(element);
            }
            if (element) element.addEventListener(event, handler);
        },

        off(element, event, handler) {
            if (typeof element === 'string') {
                element = document.getElementById(element);
            }
            if (element) element.removeEventListener(event, handler);
        },

        trigger(element, event, data) {
            if (typeof element === 'string') {
                element = document.getElementById(element);
            }
            if (element) {
                const customEvent = new CustomEvent(event, {detail: data});
                element.dispatchEvent(customEvent);
            }
        },

        delegate(parent, selector, event, handler) {
            parent.addEventListener(event, function (e) {
                if (e.target.matches(selector)) {
                    handler(e);
                }
            });
        }
    }
};

// Make utils globally available
window.utils = utils;