let model;
let isDrawing = false;

const classNames = ['', 'cat', 'airplane', 'jail', 'line', 'alarm clock', 'baseball', 'baseball bat']; // for some reason the notebook used 1-based indexing

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

ctx.strokeStyle = 'black';
ctx.lineWidth = 6;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.imageSmoothingEnabled = true;
ctx.imageSmoothingQuality = 'high';

canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
});

canvas.addEventListener('mouseup', () => {
    isDrawing = false;
});

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('result').textContent = 'Draw something';
}

function preprocessImage() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    
    tempCtx.imageSmoothingEnabled = true;
    tempCtx.imageSmoothingQuality = 'high';
    
    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, 28, 28);
    
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    
    const input = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
        const pixel = data[i * 4];
        input[i] = (255 - pixel) / 255.0;
    }
    
    return tf.tensor4d(input, [1, 28, 28, 1]);
}

async function predict() {
    if (!model) {
        document.getElementById('result').textContent = 'Model not loaded';
        return;
    }
    
    try {
        const tensor = preprocessImage();
        
        console.log('Input tensor shape:', tensor.shape);
        console.log('Input tensor min/max:', await tensor.min().data(), await tensor.max().data());
        console.log('Sample input values:', await tensor.slice([0,0,0,0], [1,5,5,1]).data());
        
        const prediction = model.predict(tensor);
        const probabilities = await prediction.data();
        
        console.log('Raw predictions:', Array.from(probabilities));
        console.log('Prediction shape:', prediction.shape);
        
        let maxIndex = 0;
        let maxProb = probabilities[0];
        
        for (let i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIndex = i;
            }
        }
        
        const confidence = (maxProb * 100).toFixed(1);
        document.getElementById('result').textContent = 
            `${classNames[maxIndex]} (${confidence}%) - Check console for debug info`;
        
        tensor.dispose();
        prediction.dispose();
        
    } catch (error) {
        console.error('Prediction error:', error);
        document.getElementById('result').textContent = 'Error: ' + error.message;
    }
}

async function loadModel() {
    try {
        model = await tf.loadGraphModel('./model/model.json');
        console.log('Model loaded');
        console.log('Input shape expected:', model.inputs[0].shape);
        console.log('Output shape:', model.outputs[0].shape);
        console.log('Model summary:', model);
        document.getElementById('result').textContent = 'Model loaded - draw something';
    } catch (error) {
        console.error('Load error:', error);
        document.getElementById('result').textContent = 'Failed to load model';
    }
}

loadModel();
