let featureExtractor;
let classifier;
let trained = false;

// Initialize ML5 Feature Extractor with MobileNet
featureExtractor = ml5.featureExtractor('MobileNet', modelReady);

function modelReady() {
    console.log('Base MobileNet Model Loaded!');
    // Create a custom classifier
    classifier = featureExtractor.classification();
    document.getElementById('trainStatus').innerText = "Status: Ready for dataset.";
}

// 1. Handling dataset input
document.getElementById('addImagesBtn').addEventListener('click', () => {
    const label = document.getElementById('labelInput').value.trim();
    const fileInput = document.getElementById('imageLoader');
    
    if (!label || fileInput.files.length === 0) {
        alert("Please enter a leaf name and select images!");
        return;
    }

    // Loop through all uploaded files for this label
    Array.from(fileInput.files).forEach(file => {
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        
        // ML5 needs the image to be loaded in the DOM briefly to process it
        img.onload = () => {
            classifier.addImage(img, label);
        };
    });

    document.getElementById('trainStatus').innerText = `Added images for: ${label}`;
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
            console.log('Loss:', lossValue); // Shows training progress
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
        alert("Train the model first!");
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
        // Display top result
        document.getElementById('result').innerText = `Result: ${results[0].label} (${Math.round(results[0].confidence * 100)}% sure)`;
    });
});
