// DOM Elements
const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const exerciseSelect = document.getElementById('exercise-select');
const durationInput = document.getElementById('exercise-duration');
const startButton = document.getElementById('start-exercise');
const resetButton = document.getElementById('reset-counter');
const timerDisplay = document.getElementById('timer');
const repCounter = document.getElementById('rep-counter');
const exerciseStage = document.getElementById('exercise-stage');
const accuracyDisplay = document.getElementById('accuracy');
const formFeedback = document.getElementById('form-feedback');
const bicepInstructions = document.getElementById('bicep-instructions');
const squatInstructions = document.getElementById('squat-instructions');

// State variables
let isExercising = false;
let exerciseInterval = null;
let timerInterval = null;
let endTime = null;

// Initialize video stream
async function initializeCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
        await videoElement.play();
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Unable to access camera. Please make sure you have granted camera permissions.');
    }
}

// Show instructions based on selected exercise
function showInstructions(exerciseType) {
    bicepInstructions.style.display = 'none';
    squatInstructions.style.display = 'none';
    
    if (exerciseType === 'bicep_curl') {
        bicepInstructions.style.display = 'block';
    } else if (exerciseType === 'squat') {
        squatInstructions.style.display = 'block';
    }
}

// Process exercise frame and update status
async function processExercise() {
    if (!isExercising) return;

    try {
        const response = await fetch('/get_exercise_status');
        if (!response.ok) {
            throw new Error('Failed to get exercise status');
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }

        // Update UI with results
        repCounter.textContent = `Reps: ${result.reps || 0}`;
        exerciseStage.textContent = `Stage: ${result.stage || 'Ready'}`;
        
        // Update accuracy display with proper fallback
        const accuracy = parseFloat(result.accuracy) || 0;
        updateAccuracy(accuracy);

        // Update form feedback with default message
        formFeedback.textContent = result.feedback || 'Get in position';
        
        // Set feedback style based on accuracy
        if (accuracy >= 96) {
            formFeedback.className = 'form-feedback success';
        } else if (accuracy >= 80) {
            formFeedback.className = 'form-feedback warning';
        } else {
            formFeedback.className = 'form-feedback error';
        }

        // Check if exercise time is up
        if (endTime && new Date().getTime() >= endTime) {
            stopExercise();
            return;
        }

        // Continue processing if still exercising
        if (isExercising) {
            setTimeout(processExercise, 100); // Poll every 100ms
        }

    } catch (error) {
        console.error('Error getting exercise status:', error);
        formFeedback.textContent = 'Error tracking exercise. Please try again.';
        formFeedback.className = 'form-feedback error';
        
        // Retry after a short delay if still exercising
        if (isExercising) {
            setTimeout(processExercise, 1000);
        }
    }
}

// Reset exercise counter
async function resetExerciseCounter() {
    try {
        const response = await fetch('/reset-counter', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const result = await response.json();
        
        if (result.success) {
            repCounter.textContent = 'Reps: 0';
            exerciseStage.textContent = 'Stage: Ready';
            formFeedback.textContent = 'Counter reset! Ready to start.';
            formFeedback.className = 'form-feedback';
        } else {
            throw new Error(result.error || 'Failed to reset counter');
        }
    } catch (error) {
        console.error('Error resetting counter:', error);
        formFeedback.textContent = 'Error resetting counter. Please try again.';
        formFeedback.className = 'form-feedback error';
    }
}

// Event Listeners
exerciseSelect.addEventListener('change', function() {
    showInstructions(this.value);
});

// Format time in MM:SS
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// Update timer display
function updateTimer() {
    if (!endTime) return;
    
    const now = new Date().getTime();
    const timeLeft = Math.max(0, (endTime - now) / 1000);
    
    if (timeLeft === 0) {
        clearInterval(timerInterval);
        stopExercise();
        return;
    }
    
    timerDisplay.textContent = `Time: ${formatTime(timeLeft)}`;
}

// Update accuracy display
function updateAccuracy(accuracy) {
    accuracyDisplay.textContent = `Accuracy: ${accuracy.toFixed(1)}%`;
    
    if (accuracy >= 96) {
        accuracyDisplay.className = 'accuracy high';
    } else if (accuracy >= 80) {
        accuracyDisplay.className = 'accuracy medium';
    } else {
        accuracyDisplay.className = 'accuracy low';
    }
}

// Stop exercise
function stopExercise() {
    isExercising = false;
    startButton.textContent = 'Start Exercise';
    clearInterval(exerciseInterval);
    clearInterval(timerInterval);
    endTime = null;
    
    // Remove video feed and show webcam preview
    const videoFeed = videoElement.previousSibling;
    if (videoFeed && videoFeed.tagName === 'IMG') {
        videoFeed.remove();
    }
    videoElement.style.display = 'block';
    
    // Reset displays
    timerDisplay.textContent = 'Time: 00:00';
    accuracyDisplay.textContent = 'Accuracy: 0%';
    accuracyDisplay.className = 'accuracy';
    repCounter.textContent = 'Reps: 0';
    exerciseStage.textContent = 'Stage: Ready';
    formFeedback.textContent = 'Exercise complete!';
    formFeedback.className = 'form-feedback';
}

startButton.addEventListener('click', async function() {
    if (!exerciseSelect.value) {
        alert('Please select an exercise first');
        return;
    }

    const duration = parseInt(durationInput.value);
    if (isNaN(duration) || duration < 30 || duration > 300) {
        alert('Please enter a valid duration between 30 and 300 seconds');
        return;
    }

    isExercising = !isExercising;
    startButton.textContent = isExercising ? 'Stop Exercise' : 'Start Exercise';
    
    if (isExercising) {
        try {
            // Clear any existing intervals
            if (exerciseInterval) clearInterval(exerciseInterval);
            if (timerInterval) clearInterval(timerInterval);

            // Reset displays
            timerDisplay.textContent = `Time: ${formatTime(duration)}`;
            repCounter.textContent = 'Reps: 0';
            exerciseStage.textContent = 'Stage: Ready';
            accuracyDisplay.textContent = 'Accuracy: 0%';
            formFeedback.textContent = 'Get ready to start!';

            // Initialize exercise with server
            const response = await fetch('/start_exercise', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    type: exerciseSelect.value,
                    duration: duration
                })
            });

            if (!response.ok) {
                throw new Error('Failed to start exercise');
            }

            // Set timer
            endTime = new Date().getTime() + (duration * 1000);
            timerInterval = setInterval(updateTimer, 1000);
            updateTimer();

            // Start the video feed
            const videoFeed = document.createElement('img');
            videoFeed.src = '/video_feed';
            videoFeed.style.width = '100%';
            videoFeed.style.height = '100%';
            videoFeed.style.objectFit = 'contain';
            videoFeed.style.display = 'block';
            
            // Wait for video feed to load
            videoFeed.onload = () => {
                videoElement.style.display = 'none';
                videoElement.parentNode.insertBefore(videoFeed, videoElement);
                
                // Start updates immediately without setInterval
                processExercise();
            };

            // Handle video feed errors
            videoFeed.onerror = () => {
                console.error('Error loading video feed');
                formFeedback.textContent = 'Error loading video feed. Please try again.';
                formFeedback.className = 'form-feedback error';
                stopExercise();
            };
        } catch (error) {
            console.error('Error starting exercise:', error);
            formFeedback.textContent = 'Failed to start exercise. Please try again.';
            formFeedback.className = 'form-feedback error';
            isExercising = false;
            startButton.textContent = 'Start Exercise';
        }
    } else {
        stopExercise();
    }
});

resetButton.addEventListener('click', resetExerciseCounter);

// Initialize camera when page loads
window.addEventListener('load', initializeCamera);