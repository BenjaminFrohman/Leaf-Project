let featureExtractor;
let classifier;
let trained = false;
let totalImagesLoaded = 0; // Counter for feedback

// Initialize ML5 Feature Extractor with MobileNet
featureExtractor = ml5.featureExtractor('MobileNet', modelReady);

function modelReady() {
    console.log('Base MobileNet Model Loaded!');
    classifier = featureExtractor.classification();
    document.getElementById('trainStatus').innerText = "Status: Ready for dataset.";
}

// 1. Handling dataset input with real-time counter feedback
document.getElementById('addImagesBtn').addEventListener('click', () => {
    const label = document.getElementById('labelInput').value.trim();
    const fileInput = document.getElementById('imageLoader');
    
    if (!label || fileInput.files.length === 0) {
        alert("Please enter a leaf name and select images!");
        return;
    }

    const filesArray = Array.from(fileInput.files);
    let sessionCount = 0;

    filesArray.forEach(file => {
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        
        img.onload = () => {
            classifier.addImage(img, label, () => {
                // This callback runs exactly when ML5 successfully processes the image
                totalImagesLoaded++;
                sessionCount++;
                
                // Update feedback text on screen
                document.getElementById('loadingFeedback').innerText = 
                    `Successfully loaded ${sessionCount} / ${filesArray.length} for "${label}" (Total Brain size: ${totalImagesLoaded})`;
            });
        };
    });

    document.getElementById('trainStatus').innerText = `Added images to queue for: ${label}`;
    fileInput.value = ''; // clear input
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
    const file = e.target.files[0];
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
        document.getElementById('result').innerText = `Result: ${results[0].label} (${Math.round(results[0].confidence * 100)}% sure)`;
    });
});

// 4. Save the brain to files
document.getElementById('saveModelBtn').addEventListener('click', () => {
    if (!trained) {
        alert("Train your model before trying to save it!");
        return;
    }
    // This prompts a download of two files: model.json and model.weights.bin
    classifier.save(); 
});

// 5. Load a saved brain back into the page
document.getElementById('loadModelBtn').addEventListener('click', () => {
    // This will prompt students to select BOTH downloaded files from their computer
    classifier.load(null, () => {
        document.getElementById('trainStatus').innerText = "Saved brain successfully loaded! Ready to test.";
        trained = true;
    });
});
