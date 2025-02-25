const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const resultsDiv = document.getElementById('results');
const ctx = canvas.getContext('2d');

//Error Handling
function handleError(error) {
  console.error("Error accessing webcam:", error);
  resultsDiv.innerHTML = "Error: Could not access webcam. Please check your camera settings and permissions.";
}


if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            video.onloadedmetadata = () => {
                video.play();
                captureAndSend();
            };
        })
        .catch(handleError);
} else {
    handleError("getUserMedia is not supported");
}

function captureAndSend() {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/jpeg');
    fetch('/classify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        resultsDiv.innerHTML = `Predicted Pose: ${data.prediction}<br>Confidence: ${data.confidence.toFixed(2)}%`;
    })
    .catch(error => {
        resultsDiv.innerHTML = `Error: ${error}`;
        console.error("Error sending image:", error);
    });
    setTimeout(captureAndSend, 500); //Adjust the delay as needed.
}