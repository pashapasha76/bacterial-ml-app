// Main app module
const App = {
    init() {
        this.bindEvents();
        this.loadModels();
    },

    bindEvents() {
        DomUtils.get('predictBtn').addEventListener('click', () => this.predict());
        DomUtils.get('imageFile').addEventListener('change', (e) => this.handleImageUpload(e));
    },

    async loadModels() {
        try {
            const modelsData = await ApiUtils.fetchModels();
            availableModels = Object.keys(modelsData);
            this.populateModelSelect();
        } catch (error) {
            console.error('Error loading models:', error);
            DomUtils.get('modelSelect').innerHTML = '<option value="">Error loading models</option>';
        }
    },

    populateModelSelect() {
        const modelSelect = DomUtils.get('modelSelect');
        
        if (availableModels.length === 0) {
            modelSelect.innerHTML = '<option value="">No models available</option>';
            return;
        }
        
        modelSelect.innerHTML = availableModels.map(modelId => 
            `<option value="${modelId}">${modelId}</option>`
        ).join('');
    },

    async handleImageUpload(event) {
        const file = event.target.files[0];
        const preview = DomUtils.get('imagePreview');
        
        if (file) {
            await ImageUtils.previewImage(file, preview);
        } else {
            preview.innerHTML = '';
            currentImageData = null;
        }
    },

    async predict() {
        const model = DomUtils.get('modelSelect').value;
        const fileInput = DomUtils.get('imageFile');
        const resultDiv = DomUtils.get('result');
        const loadingDiv = DomUtils.get('loading');
        const predictBtn = DomUtils.get('predictBtn');
        
        // Validation
        if (!model) {
            alert('Please select a model');
            return;
        }
        
        if (!fileInput.files[0]) {
            alert('Please select an image file');
            return;
        }
        
        // Show the download
        DomUtils.show(loadingDiv);
        DomUtils.disable(predictBtn);
        DomUtils.hide(resultDiv);
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        try {
            const result = await ApiUtils.predict(model, formData);
            this.displayResult(result, model, resultDiv);
        } catch (error) {
            DomUtils.show(resultDiv);
            resultDiv.innerHTML = `
                <div class="error">
                    <h3>Error</h3>
                    <p>${error.message}</p>
                </div>
            `;
        } finally {
            DomUtils.hide(loadingDiv);
            DomUtils.enable(predictBtn);
        }
    },

    displayResult(result, modelType, resultDiv) {
        DomUtils.show(resultDiv);
        
        const displayFunction = ResultDisplays[modelType] || ResultDisplays.generic;
        displayFunction(result, resultDiv);
    }
};

// Initializing app after loading DOM
document.addEventListener('DOMContentLoaded', () => App.init());