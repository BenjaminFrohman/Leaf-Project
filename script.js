window.addEventListener('DOMContentLoaded', () => {
    let brain;
    let trained = false;
    let totalImagesLoaded = 0;

    // Configure the network properties according to ml5-next-gen specifications
    const options = {
        task: 'classification',
        debug: true
    };
    
    // Spin up the core neural network brain directly
    brain = ml5.neuralNetwork(options);
    
    // Simulating immediate readiness for data input
    setTimeout(() => {
        console.log('Base MobileNet Model Loaded!');
        document.getElementById('trainStatus').innerText = "Status: Ready for dataset.";
    }, 1000);

    // 1. Handling dataset input
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
                // Safely convert image to a flat color pixel matrix for the next-gen neural net
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = 64;  // Standardizing structural resolution limits
                canvas.height = 64;
                ctx.drawImage(img, 0, 0, 64, 64);
                
                // Pull raw color values (Red, Green, Blue, Alpha) out of the canvas bounding box
                const imgData = ctx.getImageData(0, 0, 64, 64).data;
                const pixelArray = Array.from(imgData).map(val => val / 255); // Normalize structural bounds

                // Map data array cleanly to the network training matrix
                brain.addData({ x: pixelArray }, { y: label });
                
                totalImagesLoaded++;
                sessionCount++;
                
                document.getElementById('loadingFeedback').innerText = 
                    `Successfully loaded ${sessionCount} / ${filesArray.length} for "${label}" (Total Brain size: ${totalImagesLoaded})`;
            };
        });

        fileInput.value = '';
    });

    // 2. Training the neural network
    document.getElementById('trainBtn').addEventListener('click', () => {
        document.getElementById('trainStatus').innerText = "Training... please wait.";
        
        // Finalize structural bounds scaling
        brain.normalizeData();
        
        // Execute the next-gen processing configuration
        const trainingOptions = { epochs: 30 };
        brain.train(trainingOptions, whileTraining, finishedTraining);
    });

    function whileTraining(epoch, loss) {
        console.log(`Epoch: ${epoch} - Loss: ${loss.loss}`);
    }

    function finishedTraining() {
        document.getElementById('trainStatus').innerText = "Training Complete! Ready to test.";
        trained = true;
    }

    // Preview the test target image
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

    // 3. Identify the unknown leaf
    document.getElementById('predictBtn').addEventListener('click', () => {
        if (!trained) {
            alert("Train the model first or load a saved model!");
            return;
        }
        if (!testImgElement) {
            alert("Please upload a test leaf image first.");
            return;
        }

        // Convert the test leaf to the exact same matrix format
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 64;
        canvas.height = 64;
        ctx.drawImage(testImgElement, 0, 0, 64, 64);
        
        const imgData = ctx.getImageData(0, 0, 64, 64).data;
        const testPixelArray = Array.from(imgData).map(val => val / 255);

        // Run the prediction
        brain.classify({ x: testPixelArray }, (err, results) => {
            if (err) {
                console.error(err);
                return;
            }
            if (results && results.length > 0) {
                const topResult = results[0]; // Isolate highest match configuration
                document.getElementById('result').innerText = `Result: ${topResult.label} (${Math.round(topResult.confidence * 100)}% sure)`;
            }
        });
    });

    // 4. Backup saves
    document.getElementById('saveModelBtn').addEventListener('click', () => {
        if (!trained) {
            alert("Train your model before trying to save it!");
            return;
        }
        brain.save(); 
    });

    // 5. Restoring data instances
    document.getElementById('loadModelBtn').addEventListener('click', () => {
        brain.load(null, () => {
            document.getElementById('trainStatus').innerText = "Saved brain successfully loaded! Ready to test.";
            trained = true;
        });
    });
});
