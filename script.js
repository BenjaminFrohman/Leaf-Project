window.addEventListener('DOMContentLoaded', async () => {
    let classifier;
    let trained = false;
    let datasetByLabel = {};
    let testImgElement = null;
    let totalImagesLoaded = 0;
    
    console.log('Loading pre-trained model...');
    document.getElementById('trainStatus').innerText = "Loading pre-trained model...";
    
    classifier = ml5.imageClassifier('MobileNet', () => {
        console.log('Base model loaded!');
        document.getElementById('trainStatus').innerText = "Status: Ready for dataset.";
    });
    
    // === PART 1: MULTI-IMAGE DATASET INPUT ===
    document.getElementById('addImagesBtn').addEventListener('click', () => {
        const label = document.getElementById('labelInput').value.trim();
        const fileInput = document.getElementById('imageLoader');
        
        if (!label) {
            alert("Please enter a leaf name!");
            return;
        }
        if (fileInput.files.length === 0) {
            alert("Please select at least one image!");
            return;
        }
        
        if (!datasetByLabel[label]) {
            datasetByLabel[label] = 0;
        }
        
        const filesArray = Array.from(fileInput.files);
        let processedCount = 0;
        
        document.getElementById('loadingFeedback').innerText = `Processing ${filesArray.length} images...`;
        
        filesArray.forEach(file => {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            
            img.onload = () => {
                classifier.addImage(img, label);
                datasetByLabel[label]++;
                totalImagesLoaded++;
                processedCount++;
                
                if (processedCount === filesArray.length) {
                    document.getElementById('loadingFeedback').innerText = `✅ Added ${filesArray.length} images!`;
                    updateProgress();
                    fileInput.value = '';
                    document.getElementById('labelInput').value = '';
                }
            };
        });
    });
    
    function updateProgress() {
        const progressDiv = document.getElementById('datasetProgress');
        let html = '<strong>Leaves in brain:</strong><br>';
        for (const [label, count] of Object.entries(datasetByLabel)) {
            html += `${label}: ${count} photos | `;
        }
        progressDiv.innerHTML = html;
    }
    
    // === PART 2: TRAIN MODEL ===
    document.getElementById('trainBtn').addEventListener('click', async () => {
        const leafCount = Object.keys(datasetByLabel).length;
        
        if (leafCount < 2) {
            alert("Add at least 2 different leaf types before training!");
            return;
        }
        
        let totalImages = Object.values(datasetByLabel).reduce((a, b) => a + b, 0);
        
        for (const [label, count] of Object.entries(datasetByLabel)) {
            if (count < 2) {
                alert(`${label} only has ${count} photo(s). Please add at least 2 per leaf type!`);
                return;
            }
        }
        
        if (totalImages < 6) {
            alert(`You have ${totalImages} images. Try to get at least 6 total!`);
            return;
        }
        
        document.getElementById('trainStatus').innerText = "🔄 Training... this takes 5-15 seconds!";
        
        try {
            await classifier.train(function(loss) {
                console.log('Training loss:', loss);
            });
            
            finishedTraining();
        } catch (error) {
            console.error("Training error:", error);
            document.getElementById('trainStatus').innerText = "❌ Training failed. Try again!";
        }
    });
    
    function finishedTraining() {
        document.getElementById('trainStatus').innerText = "✅ Training complete! Ready to identify leaves!";
        trained = true;
    }
    
    // === PART 3: PREVIEW THE TEST IMAGE ===
    document.getElementById('testLoader').addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            const file = e.target.files[0];
            const previewDiv = document.getElementById('imagePreview');
            previewDiv.innerHTML = '';
            
            testImgElement = document.createElement('img');
            testImgElement.src = URL.createObjectURL(file);
            previewDiv.appendChild(testImgElement);
        }
    });
    
    // === PART 4: IDENTIFY LEAF ===
    document.getElementById('predictBtn').addEventListener('click', async () => {
        if (!trained) {
            alert("Train the model first!");
            return;
        }
        if (!testImgElement) {
            alert("Upload a test leaf image first!");
            return;
        }
        
        try {
            const results = await classifier.predict(testImgElement);
            console.log("Prediction results:", results);
            
            if (results && results.length > 0) {
                const topResult = results[0];
                const confidence = Math.round(topResult.confidence * 100);
                
                let message = `🎉 This looks like: <strong>${topResult.label}</strong><br>`;
                message += `I'm ${confidence}% sure!<br>`;
                
                if (confidence < 60) {
                    message += `<em>Not very confident... try uploading a clearer photo!</em>`;
                } else if (confidence < 80) {
                    message += `<em>Pretty sure, but not totally confident.</em>`;
                } else {
                    message += `<em>Very confident!</em>`;
                }
                
                document.getElementById('result').innerHTML = message;
            }
        } catch (error) {
            console.error("Prediction error:", error);
            document.getElementById('result').innerHTML = "❌ Error identifying leaf. Try again!";
        }
    });
    
    // === PART 5: SAVE / LOAD ===
    document.getElementById('saveModelBtn').addEventListener('click', () => {
        if (!trained) {
            alert("Train your model first!");
            return;
        }
        classifier.save();
        document.getElementById('trainStatus').innerText = "💾 Brain downloaded!";
    });
    
    document.getElementById('loadModelBtn').addEventListener('click', async () => {
        try {
            await classifier.load();
            document.getElementById('trainStatus').innerText = "✅ Brain loaded! Ready to identify!";
            trained = true;
        } catch (error) {
            console.error("Load error:", error);
            alert("No saved model found. Train a new one first!");
        }
    });
});