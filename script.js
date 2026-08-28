window.addEventListener('DOMContentLoaded', () => {
    let brain;
    let featureExtractor;
    let trained = false;
    let totalImagesLoaded = 0;

    // 1. Initialize the modern ml5 next-gen feature extractor
    featureExtractor = ml5.featureExtractor("MobileNet", modelReady);

    function modelReady() {
        console.log("Base MobileNet Model Loaded!");
        
        // Options configured for simple 2-category custom image mapping
        const options = {
            task: 'classification',
            architecture: 'MobileNetV2',
            debug: true
        };
        
        // Create the custom neural network brain instance
        brain = ml5.neuralNetwork(options);
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

        document.getElementById('loadingFeedback').innerText = `Processing ${filesArray.length} images...`;

        filesArray.forEach(file => {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            
            img.onload = () => {
                // Extract structural data features from image using MobileNet
                const features = featureExtractor.getFeatures(img);
                
                // Add the extracted features and user text label to the neural network
                brain.addData({ x: features }, { y: label });
                
                totalImagesLoaded++;
                sessionCount++;
                
                document.getElementById('loadingFeedback').innerText = 
                    `Successfully loaded ${sessionCount} / ${filesArray.length} for "${label}" (Total Brain size: ${totalImagesLoaded})`;
            };
        });

        fileInput.value = '';
    });

    // 3. Training the neural network
    document.getElementById('trainBtn').addEventListener('click', () => {
        document.getElementById('trainStatus').innerText = "Training... please wait.";
        
        // Organize layout structures before training execution
        brain.normalizeData();
        
        const trainingOptions = { epochs: 20 };
        brain.train(trainingOptions, whileTraining, finishedTraining);
    });

    function whileTraining(epoch, loss) {
        console.log(`Epoch: ${epoch} - Loss: ${loss.loss}`);
    }

    function finishedTraining() {
        document.getElementById('trainStatus').innerText = "Training Complete! Ready to test.";
        trained = true;
    }

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

    // 4. Identify the unknown target leaf
    document.getElementById('predictBtn').addEventListener('click', () => {
        if (!trained) {
            alert("Train the model first or load a saved model!");
            return;
        }
        if (!testImgElement) {
            alert("Please upload a test leaf image first.");
            return;
        }

        // Extract features from test target image
        const testFeatures = featureExtractor.getFeatures(testImgElement);
        
        // Pass the structural features to the trained brain for classification
        brain.classify({ x: testFeatures }, (err, results) => {
            if (err) {
                console.error(err);
                return;
            }
            if (results && results.length > 0) {
                const topResult = results[0];
                document.getElementById('result').innerText = `Result: ${topResult.label} (${Math.round(topResult.confidence * 100)}% sure)`;
            }
        });
    });

    // 5. Save the custom brain model
    document.getElementById('saveModelBtn').addEventListener('click', () => {
        if (!trained) {
            alert("Train your model before trying to save it!");
            return;
        }
        brain.save(); 
    });

    // 6. Load a saved custom brain model back into the page
    document.getElementById('loadModelBtn').addEventListener('click', () => {
        brain.load(null, () => {
            document.getElementById('trainStatus').innerText = "Saved brain successfully loaded! Ready to test.";
            trained = true;
        });
    });
});
