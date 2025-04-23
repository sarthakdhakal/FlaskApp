const video = document.getElementById('video');
const canvas = document.getElementById('capture');
const ctx = canvas.getContext('2d');
const handPreview = document.getElementById('hand-preview');
const predictionText = document.getElementById('prediction');
const audio = document.getElementById('tts-audio');

let lastHandImg = null;

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => console.error("Webcam error:", err));

function toggleHand() {
    handPreview.style.display = handPreview.style.display === "none" ? "inline-block" : "none";
}

async function captureAndPredict() {
    const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7
    });

    hands.onResults(results => {
        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const handLandmarks = results.multiHandLandmarks[0];
            const videoWidth = video.videoWidth;
            const videoHeight = video.videoHeight;

            let xMin = Math.min(...handLandmarks.map(p => p.x)) * videoWidth;
            let xMax = Math.max(...handLandmarks.map(p => p.x)) * videoWidth;
            let yMin = Math.min(...handLandmarks.map(p => p.y)) * videoHeight;
            let yMax = Math.max(...handLandmarks.map(p => p.y)) * videoHeight;

            const padding = 20;
            xMin = Math.max(xMin - padding, 0);
            yMin = Math.max(yMin - padding, 0);
            xMax = Math.min(xMax + padding, videoWidth);
            yMax = Math.min(yMax + padding, videoHeight);

            const width = xMax - xMin;
            const height = yMax - yMin;

            const cropCanvas = document.createElement('canvas');
            cropCanvas.width = width;
            cropCanvas.height = height;
            const cropCtx = cropCanvas.getContext('2d');
            cropCtx.drawImage(video, xMin, yMin, width, height, 0, 0, width, height);

            // Convert to grayscale
            const imgData = cropCtx.getImageData(0, 0, width, height);
            const data = imgData.data;
            for (let i = 0; i < data.length; i += 4) {
                const avg = 0.3 * data[i] + 0.59 * data[i + 1] + 0.11 * data[i + 2];
                data[i] = data[i + 1] = data[i + 2] = avg;
            }
            cropCtx.putImageData(imgData, 0, 0);

            // Resize and paste to hidden canvas
            ctx.clearRect(0, 0, 224, 224);
            ctx.drawImage(cropCanvas, 0, 0, 224, 224);
            const grayscaleData = canvas.toDataURL('image/jpeg');

            // Show preview
            handPreview.src = grayscaleData;

            // Send to Flask
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: grayscaleData })
            })
                .then(res => res.json())
                .then(data => {
                    predictionText.innerText = `Prediction: ${data.prediction}`;
                    audio.src = data.audio;
                    audio.load();
                    audio.play();
                })
                .catch(err => console.error("Prediction error:", err));
        } else {
            predictionText.innerText = "No hand detected!";
        }
    });

    await hands.send({ image: video });
}
