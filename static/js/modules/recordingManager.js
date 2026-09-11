// static/js/modules/recordingManager.js

import { ncaCanvas, toggleRecordingButton, recordingTimerDisplay } from './domElements.js';
import {
    state, setIsRecording, setMediaRecorder, setRecordedChunks,
    setRecordingStartTime, setRecordingTimerInterval
} from './state.js';

function chooseRecordingFormat() {
    const candidates = [
        { mimeType: 'video/mp4;codecs=avc1.42E01E', extension: 'mp4' },
        { mimeType: 'video/webm;codecs=vp9', extension: 'webm' },
        { mimeType: 'video/webm;codecs=vp8', extension: 'webm' },
        { mimeType: 'video/webm', extension: 'webm' }
    ];
    if (typeof MediaRecorder?.isTypeSupported !== 'function') return null;
    return candidates.find(candidate => MediaRecorder.isTypeSupported(candidate.mimeType)) || null;
}

export function startRecording() {
    if (typeof MediaRecorder === 'undefined' || typeof ncaCanvas.captureStream !== 'function') {
        alert('Video recording is not supported by this browser.');
        return;
    }

    setRecordedChunks([]);
    const stream = ncaCanvas.captureStream(60);
    const format = chooseRecordingFormat();

    try {
        const options = format
            ? { mimeType: format.mimeType, videoBitsPerSecond: 8_000_000 }
            : { videoBitsPerSecond: 8_000_000 };
        setMediaRecorder(new MediaRecorder(stream, options));
    } catch (error) {
        stream.getTracks().forEach(track => track.stop());
        console.error('Could not start canvas recording:', error);
        alert(`Could not start recording: ${error.message}`);
        return;
    }

    state.mediaRecorder.ondataavailable = event => {
        if (event.data.size > 0) state.recordedChunks.push(event.data);
    };

    state.mediaRecorder.onstop = () => {
        const mimeType = state.mediaRecorder.mimeType || format?.mimeType || 'video/webm';
        const extension = mimeType.includes('mp4') ? 'mp4' : 'webm';
        const blob = new Blob(state.recordedChunks, { type: mimeType });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = `canvas_video_${new Date().toISOString().replace(/[:.-]/g, '')}.${extension}`;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        URL.revokeObjectURL(url);
        stream.getTracks().forEach(track => track.stop());
    };

    state.mediaRecorder.start();
    setIsRecording(true);
    toggleRecordingButton.textContent = 'Stop Recording Video';
    toggleRecordingButton.classList.add('recording');
    startRecordingTimer();
}

export function stopRecording() {
    if (state.mediaRecorder?.state !== 'inactive') state.mediaRecorder.stop();
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
        const seconds = Math.floor((Date.now() - state.recordingStartTime) / 1000);
        recordingTimerDisplay.textContent = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
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
        if (state.isRecording) stopRecording();
        else startRecording();
    });
}
