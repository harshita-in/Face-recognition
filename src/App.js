import React, { useRef, useEffect } from "react";
import Webcam from "react-webcam";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import "@tensorflow/tfjs-backend-webgl";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  // 1. Load the MediaPipe Face Mesh model
  const loadModel = async () => {
    const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
    const detectorConfig = {
      runtime: "tfjs",
      refineLandmarks: true, // Adds detail for eyes and lips
    };
    const detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
    
    // Run detection every 40ms (~25 FPS)
    setInterval(() => {
      detect(detector);
    }, 40);
  };

  // 2. Detection logic
  const detect = async (detector) => {
    if (
      webcamRef.current &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const { videoWidth, videoHeight } = video;

      // Force canvas to match video dimensions
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      const faces = await detector.estimateFaces(video);
      
      const ctx = canvasRef.current.getContext("2d");
      requestAnimationFrame(() => drawTrace(faces, ctx));
    }
  };

  // 3. Drawing the "Trace"
  const drawTrace = (faces, ctx) => {
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    
    if (faces.length > 0) {
      faces.forEach((face) => {
        const keypoints = face.keypoints;

        // Draw the 468 landmark points
        ctx.fillStyle = "#00FF00"; // Green trace
        keypoints.forEach((keypoint) => {
          ctx.beginPath();
          ctx.arc(keypoint.x, keypoint.y, 1, 0, 2 * Math.PI);
          ctx.fill();
        });

        // Optional: Draw a bounding box around the face
        const box = face.box;
        ctx.strokeStyle = "#FF0000";
        ctx.lineWidth = 2;
        ctx.strokeRect(box.xMin, box.yMin, box.width, box.height);
      });
    }
  };

  useEffect(() => {
    loadModel();
  }, []);

  return (
    <div style={{ position: "relative", width: "100vw", height: "100vh", display: "flex", justifyContent: "center", alignItems: "center", backgroundColor: "#111" }}>
      <Webcam
        ref={webcamRef}
        muted={true}
        style={{
          position: "absolute",
          width: 640,
          height: 480,
          borderRadius: "15px",
          border: "2px solid #333"
        }}
      />
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          width: 640,
          height: 480,
          zIndex: 1
        }}
      />
      <div style={{ position: "absolute", bottom: "20px", color: "white", fontFamily: "sans-serif" }}>
        MediaPipe Real-Time Face Tracing
      </div>
    </div>
  );
}

export default App;