window.addEventListener('DOMContentLoaded', () => {
    let classifier;
    let trained = false;
    let totalImagesLoaded = 0;

    // 1. Initialize a customizable ImageClassifier instance using MobileNet
    classifier = ml5.imageClassifier('MobileNet', modelReady);

    function modelReady() {
        console.log('Base MobileNet Model Loaded!');
        document.getElementById('trainStatus').innerText = "Status: Ready for dataset.";
    }

    // 2. Handling dataset input
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
                // Modern ml5-next-gen syntax for adding custom files directly to the classifier
                classifier.addExample(img, label, () => {
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

    // 3. Training the model
    document.getElementById('trainBtn').addEventListener('click', () => {
        document.getElementById('trainStatus').innerText = "Training... please wait.";
        
        // Execute modern train command with feedback loops
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

    // 4. Identify the new leaf
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
            // Parse next-gen results array cleanly
            if (results && results.length > 0) {
                const topResult = results[0]; // Gets the highest match result
                document.getElementById('result').innerText = `Result: ${topResult.label} (${Math.round(topResult.confidence * 100)}% sure)`;
            }
        });
    });

    // 5. Save the brain to files
    document.getElementById('saveModelBtn').addEventListener('click', () => {
        if (!trained) {
            alert("Train your model before trying to save it!");
            return;
        }
        classifier.save(); 
    });

    // 6. Load a saved brain back into the page
    document.getElementById('loadModelBtn').addEventListener('click', () => {
        classifier.load(null, () => {
            document.getElementById('trainStatus').innerText = "Saved brain successfully loaded! Ready to test.";
            trained = true;
        });
    });
});
