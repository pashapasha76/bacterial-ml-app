// Functions for displaying the results of different types of models
const ResultDisplays = {
    classification(result, resultDiv) {
        const classification = result.result;
        let probabilitiesHtml = '';
        
        Object.entries(classification.all_probabilities).forEach(([className, prob]) => {
            const width = (prob * 100).toFixed(1);
            const isTop = className === classification.predicted_class;
            const barColor = isTop ? '#28a745' : '#6c757d';
            
            probabilitiesHtml += `
                <div style="margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>${className}</span>
                        <span style="font-weight: bold; color: ${isTop ? '#28a745' : '#6c757d'}">${(prob * 100).toFixed(2)}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${width}%; background: ${barColor};"></div>
                    </div>
                </div>
            `;
        });
        
        resultDiv.innerHTML = `
            <div class="success">
                <h3>Classification Results</h3>
                <div class="result-grid">
                    <div class="result-item">
                        <h4>Prediction</h4>
                        <div style="font-size: 24px; color: #28a745; font-weight: bold; margin: 10px 0;">
                            ${classification.predicted_class}
                        </div>
                        <div style="font-size: 18px; color: #666;">
                            Confidence: <strong>${(classification.confidence * 100).toFixed(2)}%</strong>
                        </div>
                    </div>
                    
                    <div class="result-item">
                        <h4>All Probabilities</h4>
                        ${probabilitiesHtml}
                    </div>
                </div>
            </div>
        `;
    },

    segmentation(result, resultDiv) {
        const segmentation = result.result;
        
        resultDiv.innerHTML = `
            <div class="success">
                <h3>Segmentation Results</h3>
                <div class="result-grid">
                    <div class="result-item">
                        <h4>Original Image</h4>
                        <img src="${currentImageData}" class="image-preview" alt="Original">
                        <div style="margin-top: 10px; font-size: 12px; color: #666;">
                            Size: ${segmentation.original_size[0]}x${segmentation.original_size[1]}
                        </div>
                    </div>
                    
                    <div class="result-item">
                        <h4>Segmentation Mask</h4>
                        <img src="data:image/png;base64,${segmentation.mask_base64}" class="mask-image" alt="Segmentation Mask">
                        <div style="margin-top: 10px; font-size: 12px; color: #666;">
                            Mask size: ${segmentation.mask_shape[0]}x${segmentation.mask_shape[1]}
                        </div>
                    </div>
                </div>
                
                <div class="stats">
                    <div class="stat-item">
                        <div>Coverage</div>
                        <div class="stat-value">${segmentation.coverage_percentage.toFixed(2)}%</div>
                    </div>
                    <div class="stat-item">
                        <div>Mask Area</div>
                        <div class="stat-value">${segmentation.mask_area.toLocaleString()} px</div>
                    </div>
                    <div class="stat-item">
                        <div>Total Pixels</div>
                        <div class="stat-value">${(segmentation.mask_shape[0] * segmentation.mask_shape[1]).toLocaleString()} px</div>
                    </div>
                </div>
            </div>
        `;
    },

    generic(result, resultDiv) {
        resultDiv.innerHTML = `
            <div class="success">
                <h3>Analysis Complete</h3>
                <pre>${JSON.stringify(result, null, 2)}</pre>
            </div>
        `;
    }
};