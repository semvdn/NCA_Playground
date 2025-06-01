// static/js/modules/recordingManager.js

import { ncaCanvas, toggleRecordingButton, recordingTimerDisplay } from './domElements.js';
import { state, setIsRecording, setMediaRecorder, setRecordedChunks, setRecordingStartTime, setRecordingTimerInterval } from './state.js';

export function startRecording() {
    setRecordedChunks([]);
    const stream = ncaCanvas.captureStream(60); // 60 FPS
    setMediaRecorder(new MediaRecorder(stream, { mimeType: 'video/mp4; codecs=avc1.42001E', videoBitsPerSecond: 20_000_000 }));

    state.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            state.recordedChunks.push(event.data);
        }
    };

    state.mediaRecorder.onstop = () => {
        const blob = new Blob(state.recordedChunks, { type: 'video/mp4' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const timestamp = new Date().toISOString().replace(/[:.-]/g, '');
        a.download = `canvas_video_${timestamp}.mp4`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url); // Clean up
    };

    state.mediaRecorder.start();
    setIsRecording(true);
    toggleRecordingButton.textContent = 'Stop Recording Video';
    toggleRecordingButton.classList.add('recording');
    startRecordingTimer();
}

export function stopRecording() {
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
        state.mediaRecorder.stop();
    }
    setIsRecording(false);
    toggleRecordingButton.textContent = 'Start Recording Video';
    toggleRecordingButton.classList.remove('recording');
    stopRecordingTimer();
}

function startRecordingTimer() {
    setRecordingStartTime(Date.now());
    recordingTimerDisplay.style.display = 'inline';
    recordingTimerDisplay.textContent = '00:00';

    if (state.recordingTimerInterval) clearInterval(state.recordingTimerInterval);
    setRecordingTimerInterval(setInterval(() => {
        const elapsedTime = Date.now() - state.recordingStartTime;
        const seconds = Math.floor(elapsedTime / 1000);
        const minutes = Math.floor(seconds / 60);
        const displaySeconds = String(seconds % 60).padStart(2, '0');
        const displayMinutes = String(minutes).padStart(2, '0');
        recordingTimerDisplay.textContent = `${displayMinutes}:${displaySeconds}`;
    }, 1000));
}

function stopRecordingTimer() {
    if (state.recordingTimerInterval) {
        clearInterval(state.recordingTimerInterval);
        setRecordingTimerInterval(null);
    }
    recordingTimerDisplay.style.display = 'none';
    recordingTimerDisplay.textContent = '00:00';
}

export function setupRecordingEvents() {
    toggleRecordingButton.addEventListener('click', () => {
        if (!state.isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });
}