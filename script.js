// Wrap everything to ensure the webpage is 100% loaded first
window.addEventListener('DOMContentLoaded', () => {
    let classifier; // In v1.4.0, the extractor itself handles everything
    let trained = false;
    let totalImagesLoaded = 0;

    // Initialize ML5 Feature Extractor directly
    classifier = ml5.featureExtractor('MobileNet', modelReady);

    function modelReady() {
        console.log('Base MobileNet Model Loaded!');
        document.getElementById('trainStatus').innerText = "Status: Ready for dataset.";
    }

    // 1. Handling dataset input with immediate visual updates
    document.getElementById('addImagesBtn').addEventListener('click', () => {
        const label = document.getElementById('labelInput').value.trim();
        const fileInput = document.getElementById('imageLoader');
        
        if (!label || fileInput.files.length === 0) {
            alert("Please enter a leaf name and select images!");
            return;
        }

        const filesArray = Array.from(fileInput.files);
        let sessionCount = 0;

        document.getElementById('loadingFeedback').innerText = `Processing ${filesArray.length} images... please wait...`;

        filesArray.forEach(file => {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            
            img.onload = () => {
                // In v1.4.0, addImage is called directly on the featureExtractor instance
                classifier.addImage(img, label, () => {
                    totalImagesLoaded++;
                    sessionCount++;
                    
                    // Update text on screen in real time
                    document.getElementById('loadingFeedback').innerText = 
                        `Successfully loaded ${sessionCount} / ${filesArray.length} for "${label}" (Total Brain size: ${totalImagesLoaded})`;
                });
            };
        });

        fileInput.value = ''; // clear file selector box
    });

    // 2. Training the model
    document.getElementById('trainBtn').addEventListener('click', () => {
        document.getElementById('trainStatus').innerText = "Training... please wait.";
        classifier.train((lossValue) => {
            if (lossValue === null) {
                document.getElementById('trainStatus').innerText = "Training Complete! Ready to test.";
                trained = true;
            } else {
                console.log('Loss:', lossValue);
            }
        });
    });

    // Preview the test image
    let testImgElement = null;
    document.getElementById('testLoader').addEventListener('change', (e) => {
        const file = e.target.files;
        if (file) {
            const previewDiv = document.getElementById('imagePreview');
            previewDiv.innerHTML = '';
            testImgElement = document.createElement('img');
            testImgElement.src = URL.createObjectURL(file);
            previewDiv.appendChild(testImgElement);
        }
    });

    // 3. Identify the new leaf
    document.getElementById('predictBtn').addEventListener('click', () => {
        if (!trained) {
            alert("Train the model first or load a saved model!");
            return;
        }
        if (!testImgElement) {
            alert("Please upload a test leaf image first.");
            return;
        }

        classifier.classify(testImgElement, (err, results) => {
            if (err) {
                console.error(err);
                return;
            }
            // v1.4.0 outputs classification arrays natively
            const topResult = results[0];
            document.getElementById('result').innerText = `Result: ${topResult.label} (${Math.round(topResult.confidence * 100)}% sure)`;
        });
    });

    // 4. Save the brain to files
    document.getElementById('saveModelBtn').addEventListener('click', () => {
        if (!trained) {
            alert("Train your model before trying to save it!");
            return;
        }
        classifier.save(); 
    });

    // 5. Load a saved brain back into the page
    document.getElementById('loadModelBtn').addEventListener('click', () => {
        classifier.load(null, () => {
            document.getElementById('trainStatus').innerText = "Saved brain successfully loaded! Ready to test.";
            trained = true;
        });
    });
});
