import React, { useRef, useEffect } from "react";
import Webcam from "react-webcam";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import "@tensorflow/tfjs-backend-webgl";
import "./App.css";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    let intervalId = null;

    const loadModel = async () => {
      const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
      const detectorConfig = {
        runtime: "tfjs",
        refineLandmarks: true, // Adds detail for eyes and lips
      };

      try {
        const detector = await faceLandmarksDetection.createDetector(model, detectorConfig);

        // Run detection every 40ms (~25 FPS)
        intervalId = setInterval(() => {
          detect(detector);
        }, 40);
      } catch (error) {
        console.error("Failed to load the model", error);
      }
    };

    const detect = async (detector) => {
      if (
        webcamRef.current &&
        webcamRef.current.video.readyState === 4 &&
        canvasRef.current
      ) {
        const video = webcamRef.current.video;
        const { videoWidth, videoHeight } = video;

        // Force canvas to match video dimensions
        webcamRef.current.video.width = videoWidth;
        webcamRef.current.video.height = videoHeight;
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;

        try {
          const faces = await detector.estimateFaces(video);
          const ctx = canvasRef.current.getContext("2d");
          requestAnimationFrame(() => drawTrace(faces, ctx));
        } catch (error) {
          console.error("Detection error:", error);
        }
      }
    };

    const drawTrace = (faces, ctx) => {
      if (!canvasRef.current) return;
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

      if (faces.length > 0) {
        faces.forEach((face) => {
          const keypoints = face.keypoints;

          // Draw the 468 landmark points
          ctx.fillStyle = "#58a6ff"; // Professional blue trace
          keypoints.forEach((keypoint) => {
            ctx.beginPath();
            ctx.arc(keypoint.x, keypoint.y, 1, 0, 2 * Math.PI);
            ctx.fill();
          });

          // Optional: Draw a bounding box around the face
          const box = face.box;
          ctx.strokeStyle = "rgba(88, 166, 255, 0.6)";
          ctx.lineWidth = 2;
          ctx.strokeRect(box.xMin, box.yMin, box.width, box.height);
        });
      }
    };

    loadModel();

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, []); // Dependencies array empty since all logic is self-contained.

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">ML Based Face Detection</h1>
        <p className="app-subtitle">Real-Time Facial Landmark Analysis</p>
      </header>

      <main className="camera-wrapper">
        <div className="camera-frame"></div>
        <Webcam
          ref={webcamRef}
          muted={true}
          className="camera-feed"
        />
        <canvas
          ref={canvasRef}
          className="camera-canvas"
        />
      </main>

      <footer className="app-footer">
        Powered by TensorFlow.js and MediaPipe
      </footer>
    </div>
  );
}

export default App;
