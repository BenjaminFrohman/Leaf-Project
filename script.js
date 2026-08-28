window.addEventListener('DOMContentLoaded', () => {
    let brain;
    let trained = false;
    let totalImagesLoaded = 0;

    const options = {
        task: 'classification',
        debug: true
    };
    
    brain = ml5.neuralNetwork(options);
    
    console.log('Base Neural Network Initialized!');
    document.getElementById('trainStatus').innerText = "Status: Ready for dataset.";

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
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = 64;  
                canvas.height = 64;
                ctx.drawImage(img, 0, 0, 64, 64);
                
                const imgData = ctx.getImageData(0, 0, 64, 64).data;
                const pixelArray = Array.from(imgData).map(val => val / 255); 

                const dataInputs = {};
                pixelArray.forEach((val, i) => {
                    dataInputs[`pixel_${i}`] = val;
                });

                const dataOutputs = {
                    label: label
                };

                brain.addData(dataInputs, dataOutputs);
                
                totalImagesLoaded++;
                sessionCount++;
                
                document.getElementById('loadingFeedback').innerText = 
                    `Successfully loaded ${sessionCount} / ${filesArray.length} for "${label}" (Total Brain size: ${totalImagesLoaded})`;
            };
        });

        fileInput.value = '';
    });

    // 2. Training the neural network
    document.getElementById('trainBtn').addEventListener('click', async () => {
        if (totalImagesLoaded === 0) {
            alert("Please add some images to the brain before training!");
            return;
        }

        document.getElementById('trainStatus').innerText = "Initializing engine... please wait.";
        
        try {
            if (ml5.tf && ml5.tf.ready) {
                await ml5.tf.ready();
            }
            
            document.getElementById('trainStatus').innerText = "Training... please wait.";
            brain.normalizeData();
            
            const trainingOptions = { epochs: 30 };
            brain.train(trainingOptions, whileTraining, finishedTraining);
        } catch (error) {
            console.error("Training engine error, attempting fallback...", error);
            brain.normalizeData();
            brain.train({ epochs: 30 }, whileTraining, finishedTraining);
        }
    });

    function whileTraining(epoch, loss) {
        console.log(`Epoch: ${epoch} - Loss: ${loss.loss}`);
    }

    function finishedTraining() {
        document.getElementById('trainStatus').innerText = "Training Complete! Ready to test.";
        trained = true;
    }

    // FIX: Preview the test image with strict target file checking to prevent 118 crashes
    let testImgElement = null;
    document.getElementById('testLoader').addEventListener('change', (e) => {
        if (e.target.value && e.target.files && e.target.files.length > 0) {
            const file = e.target.files[0]; // Fetch the singular targeted item cleanly
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

        // Run the prediction loop
        brain.classify(testInputs, (err, results) => {
            if (err) {
                console.error(err);
                return;
            }
            console.log(results); // Keeps log arrays viewable for diagnostics
            
            // FIX: Modern ml5 layouts pass results as a clean array sorted by confidence
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
