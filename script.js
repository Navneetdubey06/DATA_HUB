let currentSessionId = null;
let currentSection = 'home';
let currentDataColumns = [];
let currentDataTypes = {};
const navItems = document.querySelectorAll('.nav-item');
const sections = document.querySelectorAll('.section');
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const dataPreview = document.getElementById('dataPreview');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    setupNavigation();
    setupFileUpload();
    setupSearch();
    setupTabs();
    populateFunctions();
});

// Navigation setup
function setupNavigation() {
    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            const sectionId = this.getAttribute('data-section');
            switchSection(sectionId);
        });
    });
}

function switchSection(sectionId) {
    // Update navigation
    navItems.forEach(item => item.classList.remove('active'));
    document.querySelector(`[data-section="${sectionId}"]`).classList.add('active');

    // Update content
    sections.forEach(section => section.classList.remove('active'));
    document.getElementById(sectionId).classList.add('active');

    currentSection = sectionId;
}

// File upload setup
function setupFileUpload() {
    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    uploadStatus.innerHTML = '<div class="alert alert-success"><span class="loading"></span>Uploading and processing file...</div>';

    try {
        const response = await fetch('http://localhost:5000/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            currentSessionId = result.session_id;
            uploadStatus.innerHTML = '<div class="alert alert-success">File uploaded successfully!</div>';
            displayDataPreview(result);
        } else {
            uploadStatus.innerHTML = `<div class="alert alert-error">Error: ${result.error}</div>`;
        }
    } catch (error) {
        uploadStatus.innerHTML = `<div class="alert alert-error">Error: ${error.message}</div>`;
    }
}

function displayDataPreview(data) {
    // Store column information for parameter forms
    currentDataColumns = data.columns;
    currentDataTypes = data.dtypes;

    let html = '<h3>Data Preview</h3>';
    html += '<div class="stats-grid">';
    html += `<div class="stat-card"><div class="stat-value">${data.shape[0]}</div><div class="stat-label">Rows</div></div>`;
    html += `<div class="stat-card"><div class="stat-value">${data.shape[1]}</div><div class="stat-label">Columns</div></div>`;
    html += `<div class="stat-card"><div class="stat-value">${Object.values(data.stats).reduce((a, b) => a + (b.count || 0), 0) - data.shape[0] * data.shape[1]}</div><div class="stat-label">Missing Values</div></div>`;
    html += '</div>';

    html += '<div class="data-preview"><table class="data-table"><thead><tr>';
    data.columns.forEach(col => {
        html += `<th>${col}</th>`;
    });
    html += '</tr></thead><tbody>';
    data.preview.forEach(row => {
        html += '<tr>';
        data.columns.forEach(col => {
            html += `<td>${row[col]}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table></div>';

    dataPreview.innerHTML = html;
}

// Search setup
function setupSearch() {
    const searchInputs = ['pandasSearch', 'numpySearch', 'sklearnSearch', 'tensorflowSearch'];

    searchInputs.forEach(searchId => {
        const searchInput = document.getElementById(searchId);
        if (searchInput) {
            searchInput.addEventListener('input', function() {
                filterFunctions(searchId.replace('Search', '').toLowerCase(), this.value);
            });
        }
    });
}

function filterFunctions(library, searchTerm) {
    const functions = document.querySelectorAll(`#${library}Functions .function-card`);
    functions.forEach(card => {
        const text = card.textContent.toLowerCase();
        card.style.display = text.includes(searchTerm.toLowerCase()) ? 'block' : 'none';
    });
}

// Tab setup for visualization
function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const library = this.getAttribute('data-library');
            switchVisualizationLibrary(library);
        });
    });
}

function switchVisualizationLibrary(library) {
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[data-library="${library}"]`).classList.add('active');

    populateVisualizationFunctions(library);
}

// Populate functions
function populateFunctions() {
    populatePandasFunctions();
    populateNumpyFunctions();
    populateVisualizationFunctions('matplotlib');
    populateSklearnFunctions();
    populateTensorflowFunctions();
}

function populatePandasFunctions() {
    const functions = [
        { name: 'info', description: 'Display DataFrame information' },
        { name: 'describe', description: 'Generate descriptive statistics' },
        { name: 'head', description: 'Return the first n rows' },
        { name: 'tail', description: 'Return the last n rows' },
        { name: 'shape', description: 'Return DataFrame dimensions' },
        { name: 'columns', description: 'Get column names' },
        { name: 'dtypes', description: 'Get data types of columns' },
        { name: 'isnull', description: 'Check for missing values' },
        { name: 'corr', description: 'Compute pairwise correlation' },
        { name: 'value_counts', description: 'Count unique values' },
        { name: 'groupby', description: 'Group data and aggregate' },
        { name: 'pivot_table', description: 'Create pivot table' },
        { name: 'dropna', description: 'Remove missing values' },
        { name: 'fillna', description: 'Fill missing values' },
        { name: 'drop_duplicates', description: 'Remove duplicate rows' },
        { name: 'sort_values', description: 'Sort by values' },
        { name: 'reset_index', description: 'Reset DataFrame index' },
        { name: 'set_index', description: 'Set DataFrame index' },
        { name: 'rename', description: 'Rename columns' }
    ];

    const container = document.getElementById('pandasFunctions');
    container.innerHTML = functions.map(func =>
        `<div class="function-card" onclick="executePandasFunction('${func.name}')">
            <h4>${func.name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</h4>
            <p>${func.description}</p>
        </div>`
    ).join('');
}

function populateNumpyFunctions() {
    const functions = [
        { name: 'mean', description: 'Compute arithmetic mean' },
        { name: 'median', description: 'Compute median' },
        { name: 'std', description: 'Compute standard deviation' },
        { name: 'var', description: 'Compute variance' },
        { name: 'min', description: 'Find minimum value' },
        { name: 'max', description: 'Find maximum value' },
        { name: 'sum', description: 'Compute sum' },
        { name: 'prod', description: 'Compute product' },
        { name: 'cumsum', description: 'Compute cumulative sum' },
        { name: 'cumprod', description: 'Compute cumulative product' },
        { name: 'sort', description: 'Sort array' },
        { name: 'unique', description: 'Find unique elements' },
        { name: 'transpose', description: 'Transpose array' },
        { name: 'reshape', description: 'Reshape array' },
        { name: 'flatten', description: 'Flatten array' },
        { name: 'dot', description: 'Dot product' },
        { name: 'linalg_inv', description: 'Matrix inverse' },
        { name: 'linalg_eig', description: 'Eigenvalues/vectors' },
        { name: 'fft', description: 'Fast Fourier Transform' },
        { name: 'sin', description: 'Sine function' },
        { name: 'cos', description: 'Cosine function' },
        { name: 'exp', description: 'Exponential function' },
        { name: 'log', description: 'Natural logarithm' },
        { name: 'sqrt', description: 'Square root' }
    ];

    const container = document.getElementById('numpyFunctions');
    container.innerHTML = functions.map(func =>
        `<div class="function-card" onclick="executeNumpyFunction('${func.name}')">
            <h4>${func.name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</h4>
            <p>${func.description}</p>
        </div>`
    ).join('');
}

function populateVisualizationFunctions(library) {
    const chartTypes = {
        matplotlib: [
            { name: 'histogram', description: 'Histogram plot' },
            { name: 'scatter', description: 'Scatter plot' },
            { name: 'line', description: 'Line plot' },
            { name: 'bar', description: 'Bar chart' },
            { name: 'box', description: 'Box plot' }
        ],
        seaborn: [
            { name: 'heatmap', description: 'Correlation heatmap' },
            { name: 'pairplot', description: 'Pairwise relationships' },
            { name: 'boxplot', description: 'Box plot' },
            { name: 'violinplot', description: 'Violin plot' },
            { name: 'histplot', description: 'Histogram' }
        ],
        plotly: [
            { name: 'scatter', description: 'Interactive scatter plot' },
            { name: 'line', description: 'Interactive line plot' },
            { name: 'bar', description: 'Interactive bar chart' },
            { name: 'histogram', description: 'Interactive histogram' },
            { name: 'box', description: 'Interactive box plot' },
            { name: 'heatmap', description: 'Interactive heatmap' },
            { name: 'pie', description: 'Interactive pie chart' }
        ]
    };

    const container = document.getElementById('vizFunctions');
    container.innerHTML = chartTypes[library].map(chart =>
        `<div class="function-card" onclick="createVisualization('${library}', '${chart.name}')">
            <h4>${chart.name.replace(/\b\w/g, l => l.toUpperCase())}</h4>
            <p>${chart.description}</p>
        </div>`
    ).join('');
}

function populateSklearnFunctions() {
    const algorithms = [
        { name: 'linear_regression', description: 'Linear regression model' },
        { name: 'logistic_regression', description: 'Logistic regression for classification' },
        { name: 'decision_tree', description: 'Decision tree model' },
        { name: 'random_forest', description: 'Random forest ensemble' },
        { name: 'svm', description: 'Support Vector Machine' },
        { name: 'knn', description: 'K-Nearest Neighbors' }
    ];

    const container = document.getElementById('sklearnFunctions');
    container.innerHTML = algorithms.map(alg =>
        `<div class="function-card" onclick="executeSklearnAlgorithm('${alg.name}')">
            <h4>${alg.name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</h4>
            <p>${alg.description}</p>
        </div>`
    ).join('');
}

function populateTensorflowFunctions() {
    const models = [
        { name: 'simple_nn', description: 'Simple neural network' },
        { name: 'deep_nn', description: 'Deep neural network' },
        { name: 'cnn_1d', description: '1D Convolutional Neural Network' },
        { name: 'lstm', description: 'LSTM Recurrent Neural Network' }
    ];

    const container = document.getElementById('tensorflowFunctions');
    container.innerHTML = models.map(model =>
        `<div class="function-card" onclick="executeTensorflowModel('${model.name}')">
            <h4>${model.name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</h4>
            <p>${model.description}</p>
        </div>`
    ).join('');
}

// Function execution
async function executePandasFunction(operation) {
    if (!currentSessionId) {
        alert('Please upload data first!');
        return;
    }

    const params = await getPandasParams(operation);
    if (params === null) return;

    try {
        const response = await fetch(`http://localhost:5000/pandas/${operation}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId, params })
        });

        const result = await response.json();
        displayResult('pandasResult', result);
    } catch (error) {
        displayError('pandasResult', error.message);
    }
}

async function executeNumpyFunction(operation) {
    if (!currentSessionId) {
        alert('Please upload data first!');
        return;
    }

    // Most numpy operations don't need additional parameters
    // They work on the numeric columns of the uploaded data
    const params = {};

    // Some operations might need axis specification
    if (['mean', 'median', 'std', 'var', 'min', 'max', 'sum', 'prod', 'cumsum', 'cumprod', 'sort'].includes(operation)) {
        const axis = confirm('Calculate along columns (OK) or rows (Cancel)?');
        params.axis = axis ? 0 : 1;
    }

    try {
        const response = await fetch(`http://localhost:5000/numpy/${operation}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId, params })
        });

        const result = await response.json();
        displayResult('numpyResult', result);
    } catch (error) {
        displayError('numpyResult', error.message);
    }
}

async function createVisualization(library, chartType) {
    if (!currentSessionId) {
        alert('Please upload data first!');
        return;
    }

    const params = await getVisualizationParams(chartType);
    if (params === null) return;

    try {
        const response = await fetch(`http://localhost:5000/visualize/${library}/${chartType}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId, params })
        });

        const result = await response.json();

        if (library === 'plotly' && result.plotly_json) {
            const chartContainer = document.getElementById('chartContainer');
            chartContainer.innerHTML = '<div id="plotlyChart"></div>';
            Plotly.newPlot('plotlyChart', JSON.parse(result.plotly_json));
        } else if (result.image) {
            const chartContainer = document.getElementById('chartContainer');
            chartContainer.innerHTML = `<img src="${result.image}" style="max-width: 100%; border-radius: 10px;">`;
        }
    } catch (error) {
        displayError('chartContainer', error.message);
    }
}

async function executeSklearnAlgorithm(algorithm) {
    if (!currentSessionId) {
        alert('Please upload data first!');
        return;
    }

    const params = await getSklearnParams();
    if (params === null) return;

    try {
        const response = await fetch(`http://localhost:5000/sklearn/${algorithm}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId, params })
        });

        const result = await response.json();
        displayResult('sklearnResult', result);
    } catch (error) {
        displayError('sklearnResult', error.message);
    }
}

async function executeTensorflowModel(modelType) {
    if (!currentSessionId) {
        alert('Please upload data first!');
        return;
    }

    const params = await getTensorflowParams();
    if (params === null) return;

    try {
        const response = await fetch(`http://localhost:5000/tensorflow/${modelType}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId, params })
        });

        const result = await response.json();
        displayResult('tensorflowResult', result);
    } catch (error) {
        displayError('tensorflowResult', error.message);
    }
}

// Parameter collection functions
function getPandasParams(operation) {
    return new Promise((resolve) => {
        const params = {};
        let formHtml = '<div class="params-form"><h4>Parameters for ' + operation.replace('_', ' ') + '</h4>';

        if (operation === 'head' || operation === 'tail') {
            formHtml += `
                <div class="form-group">
                    <label>Number of rows:</label>
                    <input type="number" id="param_n" value="5" min="1" max="100">
                </div>
            `;
        } else if (operation === 'value_counts') {
            formHtml += `
                <div class="form-group">
                    <label>Select column:</label>
                    <select id="param_column" required>
                        <option value="">Choose column...</option>
                        ${currentDataColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
                    </select>
                </div>
            `;
        } else if (operation === 'groupby') {
            formHtml += `
                <div class="form-group">
                    <label>Group by column:</label>
                    <select id="param_column" required>
                        <option value="">Choose column...</option>
                        ${currentDataColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
                    </select>
                </div>
                <div class="form-group">
                    <label>Aggregation function:</label>
                    <select id="param_agg" required>
                        <option value="mean">Mean</option>
                        <option value="sum">Sum</option>
                        <option value="count">Count</option>
                        <option value="min">Min</option>
                        <option value="max">Max</option>
                        <option value="std">Standard Deviation</option>
                    </select>
                </div>
            `;
        } else if (operation === 'fillna') {
            formHtml += `
                <div class="form-group">
                    <label>Fill value:</label>
                    <input type="text" id="param_value" placeholder="Enter value to fill NaN">
                </div>
            `;
        } else if (operation === 'sort_values') {
            formHtml += `
                <div class="form-group">
                    <label>Sort by column:</label>
                    <select id="param_by">
                        <option value="">Choose column...</option>
                        ${currentDataColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
                    </select>
                </div>
                <div class="form-group">
                    <label>Sort order:</label>
                    <select id="param_ascending">
                        <option value="true">Ascending</option>
                        <option value="false">Descending</option>
                    </select>
                </div>
            `;
        } else if (operation === 'dropna') {
            formHtml += `
                <div class="form-group">
                    <label>Drop from specific columns (optional):</label>
                    <select id="param_subset" multiple>
                        ${currentDataColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
                    </select>
                    <small>Hold Ctrl/Cmd to select multiple columns</small>
                </div>
            `;
        } else if (operation === 'set_index') {
            formHtml += `
                <div class="form-group">
                    <label>Select column for index:</label>
                    <select id="param_column">
                        <option value="">Choose column...</option>
                        ${currentDataColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
                    </select>
                </div>
            `;
        } else if (operation === 'rename') {
            formHtml += `
                <div class="form-group">
                    <label>Column to rename:</label>
                    <select id="param_old_name">
                        <option value="">Choose column...</option>
                        ${currentDataColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
                    </select>
                </div>
                <div class="form-group">
                    <label>New column name:</label>
                    <input type="text" id="param_new_name" placeholder="Enter new column name">
                </div>
            `;
        } else if (operation === 'pivot_table') {
            formHtml += `
                <div class="form-group">
                    <label>Values column:</label>
                    <select id="param_values">
                        <option value="">Choose column...</option>
                        ${currentDataColumns.filter(col => currentDataTypes[col] && currentDataTypes[col].includes('int') || currentDataTypes[col].includes('float')).map(col => `<option value="${col}">${col}</option>`).join('')}
                    </select>
                </div>
                <div class="form-group">
                    <label>Index column:</label>
                    <select id="param_index">
                        <option value="">Choose column...</option>
                        ${currentDataColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
                    </select>
                </div>
            `;
        }

        if (formHtml.includes('form-group')) {
            formHtml += `
                <div style="margin-top: 20px;">
                    <button class="btn" onclick="submitParams('${operation}', 'pandas')">Execute</button>
                    <button class="btn" onclick="cancelParams()" style="background: #6c757d;">Cancel</button>
                </div>
            `;

            // Show parameter form
            const container = document.getElementById('pandasParams');
            container.innerHTML = formHtml + '</div>';

            // Store resolve function for later
            window.paramResolve = resolve;
        } else {
            // No parameters needed
            resolve(params);
        }
    });
}

function submitParams(operation, library) {
    const params = {};

    // Collect all form values
    const inputs = document.querySelectorAll('#' + library + 'Params input, #' + library + 'Params select');
    let hasErrors = false;

    inputs.forEach(input => {
        const paramName = input.id.replace('param_', '');
        let value;

        if (input.type === 'number') {
            value = parseFloat(input.value) || 0;
            if (input.hasAttribute('required') && (!input.value || isNaN(value))) {
                input.style.borderColor = '#e53e3e';
                hasErrors = true;
            } else {
                input.style.borderColor = '#e2e8f0';
            }
        } else if (input.type === 'checkbox') {
            value = input.checked;
        } else if (input.multiple) {
            const selected = Array.from(input.selectedOptions).map(option => option.value);
            value = selected.length > 0 ? selected : null;
            if (input.hasAttribute('required') && (!value || value.length === 0)) {
                input.style.borderColor = '#e53e3e';
                hasErrors = true;
            } else {
                input.style.borderColor = '#e2e8f0';
            }
        } else {
            value = input.value || null;
            if (input.hasAttribute('required') && !value) {
                input.style.borderColor = '#e53e3e';
                hasErrors = true;
            } else {
                input.style.borderColor = '#e2e8f0';
            }
        }

        params[paramName] = value;
    });

    if (hasErrors) {
        alert('Please fill in all required fields.');
        return;
    }

    // Special handling for rename operation
    if (operation === 'rename') {
        params.columns = {};
        if (params.old_name && params.new_name) {
            params.columns[params.old_name] = params.new_name;
        }
        delete params.old_name;
        delete params.new_name;
    }

    // Hide form
    document.getElementById(library + 'Params').innerHTML = '';

    // Resolve promise
    if (window.paramResolve) {
        window.paramResolve(params);
    }
}

function cancelParams() {
    // Hide all parameter forms
    ['pandasParams', 'numpyParams', 'sklearnParams', 'tensorflowParams', 'vizParams'].forEach(id => {
        document.getElementById(id).innerHTML = '';
    });

    // Resolve with null to cancel
    if (window.paramResolve) {
        window.paramResolve(null);
    }
}

function getVisualizationParams(chartType) {
    return new Promise((resolve) => {
        const params = {};
        let formHtml = '<div class="params-form"><h4>Parameters for ' + chartType + ' chart</h4>';

        if (['scatter', 'line', 'bar'].includes(chartType)) {
            formHtml += `
                <div class="form-group">
                    <label>X-axis column:</label>
                    <select id="param_x_column">
                        <option value="">Choose X column...</option>
                        ${currentDataColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
                    </select>
                </div>
                <div class="form-group">
                    <label>Y-axis column:</label>
                    <select id="param_y_column">
                        <option value="">Choose Y column...</option>
                        ${currentDataColumns.filter(col => currentDataTypes[col] && (currentDataTypes[col].includes('int') || currentDataTypes[col].includes('float'))).map(col => `<option value="${col}">${col}</option>`).join('')}
                    </select>
                </div>
            `;
        } else if (chartType === 'histogram') {
            formHtml += `
                <div class="form-group">
                    <label>Select column:</label>
                    <select id="param_column">
                        <option value="">Choose column...</option>
                        ${currentDataColumns.filter(col => currentDataTypes[col] && (currentDataTypes[col].includes('int') || currentDataTypes[col].includes('float'))).map(col => `<option value="${col}">${col}</option>`).join('')}
                    </select>
                </div>
            `;
        } else if (chartType === 'box') {
            formHtml += `
                <div class="form-group">
                    <label>Select column:</label>
                    <select id="param_column">
                        <option value="">Choose column...</option>
                        ${currentDataColumns.filter(col => currentDataTypes[col] && (currentDataTypes[col].includes('int') || currentDataTypes[col].includes('float'))).map(col => `<option value="${col}">${col}</option>`).join('')}
                    </select>
                </div>
            `;
        } else if (chartType === 'pie') {
            formHtml += `
                <div class="form-group">
                    <label>Select column:</label>
                    <select id="param_column">
                        <option value="">Choose column...</option>
                        ${currentDataColumns.filter(col => currentDataTypes[col] && currentDataTypes[col].includes('object')).map(col => `<option value="${col}">${col}</option>`).join('')}
                    </select>
                </div>
            `;
        } else if (chartType === 'heatmap') {
            // No additional parameters needed for correlation heatmap
        } else if (chartType === 'pairplot') {
            // No additional parameters needed
        }

        if (formHtml.includes('form-group')) {
            formHtml += `
                <div style="margin-top: 20px;">
                    <button class="btn" onclick="submitVizParams('${chartType}')">Create Chart</button>
                    <button class="btn" onclick="cancelParams()" style="background: #6c757d;">Cancel</button>
                </div>
            `;

            // Show parameter form
            const container = document.getElementById('vizParams');
            container.innerHTML = formHtml + '</div>';

            // Store resolve function for later
            window.paramResolve = resolve;
        } else {
            // No parameters needed
            resolve(params);
        }
    });
}

function submitVizParams(chartType) {
    const params = {};

    // Collect all form values
    const inputs = document.querySelectorAll('#vizParams input, #vizParams select');
    inputs.forEach(input => {
        const paramName = input.id.replace('param_', '');
        if (input.multiple) {
            const selected = Array.from(input.selectedOptions).map(option => option.value);
            params[paramName] = selected.length > 0 ? selected : null;
        } else {
            params[paramName] = input.value || null;
        }
    });

    // Hide form
    document.getElementById('vizParams').innerHTML = '';

    // Resolve promise
    if (window.paramResolve) {
        window.paramResolve(params);
    }
}

function getSklearnParams() {
    return new Promise((resolve) => {
        let formHtml = '<div class="params-form"><h4>Parameters for Machine Learning</h4>';
        formHtml += `
            <div class="form-group">
                <label>Target column (for supervised learning):</label>
                <select id="param_target_column" required>
                    <option value="">Choose target column...</option>
                    ${currentDataColumns.map(col => `<option value="${col}">${col} (${currentDataTypes[col]})</option>`).join('')}
                </select>
            </div>
            <div style="margin-top: 20px;">
                <button class="btn" onclick="submitSklearnParams()">Execute Algorithm</button>
                <button class="btn" onclick="cancelParams()" style="background: #6c757d;">Cancel</button>
            </div>
        `;

        // Show parameter form
        const container = document.getElementById('sklearnParams');
        container.innerHTML = formHtml + '</div>';

        // Store resolve function for later
        window.paramResolve = resolve;
    });
}

function submitSklearnParams() {
    const params = {};

    // Collect all form values
    const inputs = document.querySelectorAll('#sklearnParams input, #sklearnParams select');
    inputs.forEach(input => {
        const paramName = input.id.replace('param_', '');
        params[paramName] = input.value || null;
    });

    // Hide form
    document.getElementById('sklearnParams').innerHTML = '';

    // Resolve promise
    if (window.paramResolve) {
        window.paramResolve(params);
    }
}

function getTensorflowParams() {
    return new Promise((resolve) => {
        let formHtml = '<div class="params-form"><h4>Parameters for Deep Learning</h4>';
        formHtml += `
            <div class="form-group">
                <label>Target column:</label>
                <select id="param_target_column" required>
                    <option value="">Choose target column...</option>
                    ${currentDataColumns.filter(col => currentDataTypes[col] && (currentDataTypes[col].includes('int') || currentDataTypes[col].includes('float'))).map(col => `<option value="${col}">${col} (${currentDataTypes[col]})</option>`).join('')}
                </select>
            </div>
            <div class="form-group">
                <label>Training epochs:</label>
                <input type="number" id="param_epochs" value="50" min="1" max="500">
            </div>
            <div class="form-group">
                <label>Batch size:</label>
                <input type="number" id="param_batch_size" value="32" min="1" max="512">
            </div>
            <div style="margin-top: 20px;">
                <button class="btn" onclick="submitTensorflowParams()">Train Model</button>
                <button class="btn" onclick="cancelParams()" style="background: #6c757d;">Cancel</button>
            </div>
        `;

        // Show parameter form
        const container = document.getElementById('tensorflowParams');
        container.innerHTML = formHtml + '</div>';

        // Store resolve function for later
        window.paramResolve = resolve;
    });
}

function submitTensorflowParams() {
    const params = {};

    // Collect all form values
    const inputs = document.querySelectorAll('#tensorflowParams input, #tensorflowParams select');
    inputs.forEach(input => {
        const paramName = input.id.replace('param_', '');
        if (input.type === 'number') {
            params[paramName] = parseInt(input.value) || 32;
        } else {
            params[paramName] = input.value || null;
        }
    });

    // Hide form
    document.getElementById('tensorflowParams').innerHTML = '';

    // Resolve promise
    if (window.paramResolve) {
        window.paramResolve(params);
    }
}

// Result display
function displayResult(containerId, result) {
    const container = document.getElementById(containerId);
    let html = '<div class="result-area"><h4>Result</h4><div class="result-content">';

    if (result.result) {
        if (typeof result.result === 'object') {
            html += JSON.stringify(result.result, null, 2);
        } else {
            html += result.result;
        }
    } else {
        html += JSON.stringify(result, null, 2);
    }

    html += '</div></div>';
    container.innerHTML = html;
}

function displayError(containerId, error) {
    const container = document.getElementById(containerId);
    container.innerHTML = `<div class="alert alert-error">Error: ${error}</div>`;
}

// AI Summary
document.getElementById('generateSummary').addEventListener('click', async function() {
    if (!currentSessionId) {
        alert('Please upload data first!');
        return;
    }

    console.log('AI Summary button clicked');
    console.log('Current session ID:', currentSessionId);

    this.innerHTML = '<span class="loading"></span> Generating...';
    this.disabled = true;

    try {
        console.log('Sending request to AI summary endpoint...');
        const response = await fetch('http://localhost:5000/ai_summary', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId })
        });

        console.log('Response status:', response.status);
        const result = await response.json();
        console.log('Response result:', result);

        const container = document.getElementById('aiSummaryResult');
        if (result.summary) {
            container.innerHTML = `<div class="result-area"><h4>AI Summary</h4><div class="result-content">${result.summary}</div></div>`;
        } else if (result.error) {
            container.innerHTML = `<div class="alert alert-error">Error: ${result.error}</div>`;
        } else {
            container.innerHTML = `<div class="alert alert-error">Unknown error occurred</div>`;
        }
    } catch (error) {
        console.error('Frontend error:', error);
        displayError('aiSummaryResult', error.message);
    } finally {
        this.innerHTML = '<i class="fas fa-magic"></i> Generate AI Summary';
        this.disabled = false;
    }
});

// Download
document.getElementById('downloadBtn').addEventListener('click', function() {
    if (!currentSessionId) {
        alert('Please upload data first!');
        return;
    }

    window.open(`http://localhost:5000/download/${currentSessionId}`, '_blank');
});