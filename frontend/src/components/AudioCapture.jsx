import React, { useEffect, useRef, useState } from 'react';

const AudioCapture = ({ active, onMetrics }) => {
  const [isRecording, setIsRecording] = useState(false);
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (active) {
      startCapture();
    } else {
      stopCapture();
    }
    return () => stopCapture();
  }, [active]);

  const startCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setIsRecording(true);

      // Connect WS
      const ws = new WebSocket('ws://localhost:8000/ws/audio');
      wsRef.current = ws;

      ws.onmessage = (event) => {
        const metrics = JSON.parse(event.data);
        if (onMetrics) onMetrics(metrics);
      };

      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioCtx;

      const input = audioCtx.createMediaStreamSource(stream);
      inputRef.current = input;

      // Create ScriptProcessorNode to capture raw PCM
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (ws.readyState === WebSocket.OPEN) {
          const inputData = e.inputBuffer.getChannelData(0);
          // Convert float32 array to 16-bit PCM bytes
          const pcmData = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7fff;
          }
          ws.send(pcmData.buffer);
        }
      };

      input.connect(processor);
      processor.connect(audioCtx.destination);
    } catch (err) {
      console.error('Audio capture initialization error:', err);
      setIsRecording(false);
    }
  };

  const stopCapture = () => {
    if (processorRef.current && inputRef.current) {
      inputRef.current.disconnect();
      processorRef.current.disconnect();
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    if (wsRef.current) {
      wsRef.current.close();
    }
    setIsRecording(false);
  };

  return (
    <div className="audio-capture-widget p-3 rounded-lg border border-gray-700 bg-gray-900 text-white">
      <div className="flex items-center space-x-2">
        <span className={`w-3 h-3 rounded-full ${isRecording ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
        <span className="text-sm font-medium">
          {isRecording ? 'Microphone Active (Streaming to Agent)' : 'Microphone Inactive'}
        </span>
      </div>
    </div>
  );
};

export default AudioCapture;
