window.addEventListener('DOMContentLoaded', () => {
    let brain;
    let trained = false;
    let datasetByLabel = {}; // Track images per leaf type
    let testImgElement = null;
    
    const options = {
        task: 'classification',
        debug: false
    };
    
    brain = ml5.neuralNetwork(options);
    console.log('Neural Network Ready!');
    
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
        
        // Initialize this leaf type if new
        if (!datasetByLabel[label]) {
            datasetByLabel[label] = 0;
        }
        
        const filesArray = Array.from(fileInput.files);
        let processedCount = 0;
        
        document.getElementById('loadingFeedback').innerText = `Processing ${filesArray.length} images...`;
        
        filesArray.forEach(file => {
            const reader = new FileReader();
            reader.onload = (event) => {
                const img = new Image();
                img.onload = () => {
                    // Resize to 64x64
                    const canvas = document.createElement('canvas');
                    canvas.width = 64;
                    canvas.height = 64;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, 64, 64);
                    
                    // Convert to pixel array
                    const imgData = ctx.getImageData(0, 0, 64, 64).data;
                    const pixelArray = Array.from(imgData).map(val => val / 255);
                    
                    // Create input object
                    const inputs = {};
                    pixelArray.forEach((val, i) => {
                        inputs[`pixel_${i}`] = val;
                    });
                    
                    // Add to brain
                    brain.addData(inputs, { label: label });
                    datasetByLabel[label]++;
                    processedCount++;
                    
                    // Update UI when done
                    if (processedCount === filesArray.length) {
                        document.getElementById('loadingFeedback').innerText = `✅ Added ${filesArray.length} images!`;
                        updateProgress();
                        fileInput.value = '';
                        document.getElementById('labelInput').value = '';
                    }
                };
                img.src = event.target.result;
            };
            reader.readAsDataURL(file);
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
        if (totalImages < 10) {
            alert(`You have ${totalImages} images. Try to get at least 10 total!`);
            return;
        }
        
        document.getElementById('trainStatus').innerText = "🔄 Training... this takes 10-30 seconds!";
        
        try {
            brain.normalizeData();
            brain.train({ epochs: 30 }, whileTraining, finishedTraining);
        } catch (error) {
            console.error("Training error:", error);
            brain.normalizeData();
            brain.train({ epochs: 250 }, whileTraining, finishedTraining);
        }
    });
    
    function whileTraining(epoch, loss) {
        if (epoch % 5 === 0) {
            console.log(`Epoch ${epoch}: Loss = ${loss.loss.toFixed(3)}`);
        }
    }
    
    function finishedTraining() {
        document.getElementById('trainStatus').innerText = "✅ Training complete! Ready to identify leaves!";
        trained = true;
    }
    
    // === PART 3: TEST IMAGE PREVIEW ===
    document.getElementById('testLoader').addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            const file = e.target.files[0];
            const previewDiv = document.getElementById('imagePreview');
            previewDiv.innerHTML = '';
            
            const reader = new FileReader();
            reader.onload = (event) => {
                testImgElement = document.createElement('img');
                testImgElement.src = event.target.result;
                previewDiv.appendChild(testImgElement);
            };
            reader.readAsDataURL(file);
        }
    });
    
    // === PART 4: IDENTIFY LEAF ===
    document.getElementById('predictBtn').addEventListener('click', () => {
        if (!trained) {
            alert("Train the model first!");
            return;
        }
        if (!testImgElement) {
            alert("Upload a test leaf image first!");
            return;
        }
        
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 64;
        canvas.height = 64;
        ctx.drawImage(testImgElement, 0, 0, 64, 64);
        
        const imgData = ctx.getImageData(0, 0, 64, 64).data;
        const testPixelArray = Array.from(imgData).map(val => val / 255);
        
        const testInputs = {};
        testPixelArray.forEach((val, i) => {
            testInputs[`pixel_${i}`] = val;
        });
        
        brain.classify(testInputs, (err, results) => {
            if (err) {
                console.error(err);
                return;
            }
            
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
        });
    });
    
    // === PART 5: SAVE / LOAD ===
    document.getElementById('saveModelBtn').addEventListener('click', () => {
        if (!trained) {
            alert("Train your model first!");
            return;
        }
        brain.save();
        document.getElementById('trainStatus').innerText = "💾 Brain downloaded!";
    });
    
    document.getElementById('loadModelBtn').addEventListener('click', () => {
        brain.load(null, () => {
            document.getElementById('trainStatus').innerText = "✅ Brain loaded! Ready to identify!";
            trained = true;
            document.getElementById('result').innerHTML = '';
        });
    });
});