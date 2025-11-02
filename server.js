// backend/server.js
const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const bodyParser = require("body-parser");
const cors = require("cors");
const fs = require("fs");

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*" }
});

app.use(cors());
// Keep URL-encoded parsing for any form-like routes you might add later
app.use(bodyParser.urlencoded({ extended: true }));

// In-memory store (simple). Replace with DB in production.
let latestPerCamera = {};
let events = []; // recent events

// ---------- JSON route (existing) ----------
// POST /detect   -> expects JSON { cameraId, imageBase64, object?, confidence? }
app.post("/detect", express.json({ limit: "12mb" }), (req, res) => {
  try {
    const { cameraId, imageBase64, object, confidence } = req.body || {};
    if (!cameraId || !imageBase64) return res.status(400).json({ error: "cameraId & imageBase64 required" });

    const timestamp = new Date().toISOString();
    const event = { cameraId, imageBase64, object: object || null, confidence: confidence || null, timestamp };

    latestPerCamera[cameraId] = event;
    events.unshift(event);
    if (events.length > 200) events.pop();

    io.emit("detection", event);
    return res.json({ status: "ok", timestamp });
  } catch (err) {
    console.error("JSON /detect error:", err);
    return res.status(500).json({ error: err.toString() });
  }
});

// ---------- Raw JPG route (binary) ----------
// POST /detect-binary?cameraId=cam01  -> expects Content-Type: image/jpeg, body = raw jpeg bytes
app.post("/detect-binary", express.raw({ type: "image/*", limit: "12mb" }), (req, res) => {
  try {
    const cameraId = (req.query.cameraId || req.headers["camera-id"] || "unknown").toString();
    if (!cameraId) return res.status(400).json({ error: "cameraId query param or camera-id header required" });

    const imgBuf = req.body;
    if (!imgBuf || !imgBuf.length) return res.status(400).json({ error: "empty body" });

    // convert raw buffer to base64 data URI so front-end can show it easily
    const b64 = imgBuf.toString("base64");
    const dataUri = `data:image/jpeg;base64,${b64}`;

    const timestamp = new Date().toISOString();
    const event = { cameraId, imageBase64: dataUri, object: null, confidence: null, timestamp };

    latestPerCamera[cameraId] = event;
    events.unshift(event);
    if (events.length > 200) events.pop();

    io.emit("detection", event);
    return res.json({ status: "ok", cameraId, timestamp });
  } catch (err) {
    console.error("Binary /detect-binary error:", err);
    return res.status(500).json({ error: err.toString() });
  }
});

// Simple API to get current state
app.get("/state", (req, res) => {
  res.json({ latestPerCamera, recentEvents: events.slice(0, 50) });
});

// optional: small healthcheck
app.get("/health", (req, res) => res.json({ status: "ok", time: new Date().toISOString() }));

// start server
const PORT = process.env.PORT || 3000;
server.listen(PORT, '0.0.0.0', () => {
  console.log(`Backend listening on ${PORT}`);
  console.log(`Access from ESP32: http://10.118.99.68:${PORT}`);
});

app.get("/", (req, res) => {
  res.sendFile(__dirname + "/frontend.html");
});

// socket.io connection logging
io.on("connection", (socket) => {
  console.log("socket connected:", socket.id);
  // send current state on connect
  socket.emit("init", { latestPerCamera, recentEvents: events.slice(0, 50) });
  socket.on("disconnect", () => console.log("socket disconnected:", socket.id));
});
