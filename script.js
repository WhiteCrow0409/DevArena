const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const poseSelect = document.getElementById('poseSelect');
const startButton = document.getElementById('startButton');
const timerDiv = document.getElementById('timer');
const countdown = document.getElementById('countdown');
const accuracyBar = document.getElementById('accuracyBar');
const accuracyText = document.getElementById('accuracyText');
const results = document.getElementById('results');
const finalScore = document.getElementById('finalScore');
const feedback = document.getElementById('feedback');

let stream = null;
let isEvaluating = false;
let evaluationTimer = null;
let averageAccuracy = 0;
let totalReadings = 0;
let selectedPose = '';

// Enable start button when pose is selected
poseSelect.addEventListener('change', (e) => {
    selectedPose = e.target.value;
    startButton.disabled = !selectedPose;
});

// Handle start button click
startButton.addEventListener('click', async () => {
    if (!stream) {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            await video.play();
        } catch (error) {
            handleError("Error accessing webcam: " + error.message);
            return;
        }
    }

    startEvaluation();
});

function startEvaluation() {
    // Reset values
    averageAccuracy = 0;
    totalReadings = 0;
    isEvaluating = true;
    
    // Show/hide elements
    timerDiv.classList.remove('d-none');
    results.classList.add('d-none');
    startButton.disabled = true;
    poseSelect.disabled = true;
    
    // Start 15-second timer
    let timeLeft = 15;
    countdown.textContent = timeLeft;
    
    const timer = setInterval(() => {
        timeLeft--;
        countdown.textContent = timeLeft;
        
        if (timeLeft <= 0) {
            clearInterval(timer);
            endEvaluation();
        }
    }, 1000);
    
    // Start continuous evaluation
    evaluateFrame();
}

function evaluateFrame() {
    if (!isEvaluating) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/jpeg');

    fetch('/classify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
            image: imageData,
            expected_pose: selectedPose
        })
    })
    .then(response => response.json())
    .then(data => {
        if (isEvaluating) {
            updateAccuracy(data.confidence);
            if (data.prediction !== selectedPose) {
                accuracyText.textContent = `Accuracy: 0% (Wrong Pose: ${data.prediction})`;
            } else {
                accuracyText.textContent = `Accuracy: ${data.confidence.toFixed(1)}%`;
            }
            accuracyBar.style.width = `${data.confidence}%`;
            requestAnimationFrame(evaluateFrame);
        }
    })
    .catch(error => {
        console.error("Error during evaluation:", error);
        if (isEvaluating) {
            requestAnimationFrame(evaluateFrame);
        }
    });
}

function updateAccuracy(confidence) {
    totalReadings++;
    averageAccuracy = (averageAccuracy * (totalReadings - 1) + confidence) / totalReadings;
}

function endEvaluation() {
    isEvaluating = false;
    
    // Reset UI
    timerDiv.classList.add('d-none');
    results.classList.remove('d-none');
    startButton.disabled = false;
    poseSelect.disabled = false;
    
    // Show final results
    const finalAccuracy = averageAccuracy.toFixed(1);
    finalScore.textContent = `Final Score: ${finalAccuracy}%`;
    
    // Provide feedback based on score
    if (finalAccuracy >= 90) {
        feedback.textContent = "Excellent! Your pose form is perfect!";
    } else if (finalAccuracy >= 75) {
        feedback.textContent = "Great job! Keep practicing to perfect your form.";
    } else if (finalAccuracy >= 60) {
        feedback.textContent = "Good effort! Try to maintain proper alignment.";
    } else {
        feedback.textContent = "Keep practicing! Focus on the correct pose form.";
    }
}

function handleError(error) {
    console.error(error);
    feedback.textContent = error;
    feedback.style.color = '#dc3545';
    results.classList.remove('d-none');
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});