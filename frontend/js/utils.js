// Global variables
let currentImageData = null;
let availableModels = [];

// Utilities for working with DOM
const DomUtils = {
    get: (id) => document.getElementById(id),
    show: (element) => element.style.display = 'block',
    hide: (element) => element.style.display = 'none',
    enable: (element) => element.disabled = false,
    disable: (element) => element.disabled = true
};

// Utilities for API
const ApiUtils = {
    async fetchModels() {
        const response = await fetch('/models');
        if (!response.ok) throw new Error('Failed to load models');
        return await response.json();
    },
    
    async predict(model, formData) {
        const response = await fetch(`/predict/${model}`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error('Prediction failed');
        return await response.json();
    }
};

// Utilities for working with images
const ImageUtils = {
    previewImage(file, previewElement) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = function(e) {
                currentImageData = e.target.result;
                previewElement.innerHTML = `<img src="${e.target.result}" class="image-preview" alt="Preview">`;
                resolve(e.target.result);
            };
            reader.readAsDataURL(file);
        });
    }
};