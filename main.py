from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from datetime import datetime
import logging
import sys
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('flask_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Load Models
logger.info("Loading AI models...")

# 1. YOLO for object detection
try:
    yolo_model = YOLO('yolov8n.pt')
    logger.info("✅ YOLOv8 loaded")
except Exception as e:
    logger.error(f"❌ Failed to load YOLO: {e}")
    yolo_model = None

# 2. BLIP for image captioning (describes the entire scene)
try:
    logger.info("Loading BLIP model (this may take a minute)...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    logger.info("✅ BLIP loaded (image captioning)")
except Exception as e:
    logger.error(f"❌ Failed to load BLIP: {e}")
    blip_processor = None
    blip_model = None

logger.info("="*60)
logger.info("Models loaded successfully!")
logger.info("="*60)

# Store latest detections
latest_per_camera = {}
events = []
request_count = 0

def log_request_info(route_name):
    """Log detailed request information"""
    global request_count
    request_count += 1
    
    logger.info(f"{'='*60}")
    logger.info(f"[Request #{request_count}] {route_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Method: {request.method}")
    logger.info(f"URL: {request.url}")
    logger.info(f"Remote Address: {request.remote_addr}")
    logger.info(f"Content-Length: {request.content_length}")

def generate_image_caption(image_pil):
    """Generate natural language description of image using BLIP"""
    try:
        if blip_model is None or blip_processor is None:
            return None
        
        logger.info("Generating image caption with BLIP...")
        
        # Process image
        inputs = blip_processor(image_pil, return_tensors="pt")
        
        # Generate caption
        out = blip_model.generate(**inputs, max_length=50)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        
        logger.info(f"Generated caption: {caption}")
        return caption
        
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        return None

def process_image_advanced(image_data, source="unknown"):
    """Process image with multiple AI models"""
    try:
        logger.info(f"Processing image from {source}")
        logger.info(f"Image data length: {len(image_data)}")
        
        # Convert to numpy array for OpenCV
        nparr = np.frombuffer(image_data, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv is None:
            logger.error("Failed to decode image")
            return None
        
        logger.info(f"Decoded image shape: {img_cv.shape}")
        
        # Convert to PIL for BLIP
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        result = {
            'yolo_detections': [],
            'scene_description': None,
            'summary': None
        }
        
        # 1. YOLO Detection
        if yolo_model is not None:
            logger.info("Running YOLO detection...")
            yolo_results = yolo_model(img_cv, verbose=False)
            
            annotated_img = img_cv.copy()
            
            for yolo_result in yolo_results:
                boxes = yolo_result.boxes
                logger.info(f"YOLO found {len(boxes)} objects")
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = yolo_model.names[class_id]
                    
                    result['yolo_detections'].append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
                    
                    # Draw on image
                    color = (0, 255, 0)
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # Background for text
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        annotated_img, 
                        (x1, y1 - text_height - 10), 
                        (x1 + text_width, y1), 
                        color, 
                        -1
                    )
                    cv2.putText(
                        annotated_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
                    )
            
            # Convert annotated image to base64
            _, buffer = cv2.imencode('.jpg', annotated_img)
            annotated_b64 = base64.b64encode(buffer).decode('utf-8')
            result['annotated_image'] = f"data:image/jpeg;base64,{annotated_b64}"
        else:
            # No annotation, return original
            original_b64 = base64.b64encode(image_data).decode('utf-8')
            result['annotated_image'] = f"data:image/jpeg;base64,{original_b64}"
        
        # 2. BLIP Image Captioning (describes entire scene)
        caption = generate_image_caption(img_pil)
        if caption:
            result['scene_description'] = caption
        
        # 3. Generate human-readable summary
        summary_parts = []
        
        if result['yolo_detections']:
            object_counts = {}
            for det in result['yolo_detections']:
                obj = det['class']
                object_counts[obj] = object_counts.get(obj, 0) + 1
            
            obj_list = [f"{count} {obj}{'s' if count > 1 else ''}" 
                       for obj, count in object_counts.items()]
            summary_parts.append(f"Detected: {', '.join(obj_list)}")
        
        if result['scene_description']:
            summary_parts.append(f"Scene: {result['scene_description']}")
        
        result['summary'] = ". ".join(summary_parts) if summary_parts else "No objects detected"
        
        logger.info(f"✅ Processing complete")
        logger.info(f"   YOLO objects: {len(result['yolo_detections'])}")
        logger.info(f"   Scene description: {result['scene_description']}")
        logger.info(f"   Summary: {result['summary']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return None

@app.route('/health', methods=['GET'])
def health():
    log_request_info("HEALTH CHECK")
    health_info = {
        'status': 'ok',
        'time': datetime.now().isoformat(),
        'models': {
            'yolo': yolo_model is not None,
            'blip': blip_model is not None
        },
        'total_requests': request_count,
        'total_events': len(events)
    }
    logger.info(f"Health check response: {health_info}")
    return jsonify(health_info)

@app.route('/detect-binary', methods=['POST'])
def detect_binary():
    log_request_info("BINARY DETECTION")
    
    try:
        camera_id = request.args.get('cameraId', 'unknown')
        logger.info(f"Camera ID: {camera_id}")
        
        image_data = request.get_data()
        
        if not image_data:
            logger.error("No image data received")
            return jsonify({'error': 'No image data'}), 400
        
        logger.info(f"Received {len(image_data)} bytes")
        
        # Check JPEG
        if len(image_data) >= 2:
            is_jpeg = image_data[0] == 0xFF and image_data[1] == 0xD8
            logger.info(f"Is valid JPEG: {is_jpeg}")
        
        # Save debug image
        debug_filename = f"debug_{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        try:
            with open(debug_filename, 'wb') as f:
                f.write(image_data)
            logger.info(f"Saved to: {debug_filename}")
        except Exception as e:
            logger.error(f"Failed to save: {e}")
        
        # Process with advanced AI
        logger.info("Starting advanced AI processing...")
        analysis = process_image_advanced(image_data, camera_id)
        
        if analysis is None:
            logger.error("AI processing failed")
            original_b64 = base64.b64encode(image_data).decode('utf-8')
            analysis = {
                'yolo_detections': [],
                'scene_description': None,
                'summary': 'Processing failed',
                'annotated_image': f"data:image/jpeg;base64,{original_b64}"
            }
        
        # Create event
        timestamp = datetime.now().isoformat()
        event = {
            'cameraId': camera_id,
            'imageBase64': analysis['annotated_image'],
            'detections': analysis['yolo_detections'],
            'sceneDescription': analysis['scene_description'],
            'summary': analysis['summary'],
            'object': analysis['yolo_detections'][0]['class'] if analysis['yolo_detections'] else None,
            'confidence': analysis['yolo_detections'][0]['confidence'] if analysis['yolo_detections'] else None,
            'timestamp': timestamp,
            'size': len(image_data)
        }
        
        # Store event
        latest_per_camera[camera_id] = event
        events.insert(0, event)
        if len(events) > 200:
            events.pop()
        
        logger.info(f"✅ Detection complete for {camera_id}")
        logger.info(f"   Summary: {event['summary']}")
        
        # Emit to Socket.IO clients
        try:
            socketio.emit('detection', event)
            logger.info(f"Emitted detection event")
        except Exception as e:
            logger.error(f"Socket.IO emit failed: {e}")
        
        return jsonify({
            'status': 'ok',
            'cameraId': camera_id,
            'timestamp': timestamp,
            'detections': len(analysis['yolo_detections']),
            'sceneDescription': analysis['scene_description'],
            'summary': analysis['summary'],
            'saved_to': debug_filename
        })
        
    except Exception as e:
        logger.error(f"Error in detect_binary: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect_json():
    log_request_info("JSON DETECTION")
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data'}), 400
        
        camera_id = data.get('cameraId', 'unknown')
        image_base64 = data.get('imageBase64', '')
        
        if not image_base64:
            return jsonify({'error': 'No imageBase64'}), 400
        
        # Decode base64
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        
        # Process
        analysis = process_image_advanced(image_data, camera_id)
        
        if analysis is None:
            return jsonify({'error': 'Processing failed'}), 500
        
        timestamp = datetime.now().isoformat()
        event = {
            'cameraId': camera_id,
            'imageBase64': analysis['annotated_image'],
            'detections': analysis['yolo_detections'],
            'sceneDescription': analysis['scene_description'],
            'summary': analysis['summary'],
            'timestamp': timestamp
        }
        
        latest_per_camera[camera_id] = event
        events.insert(0, event)
        if len(events) > 200:
            events.pop()
        
        socketio.emit('detection', event)
        
        return jsonify({
            'status': 'ok',
            'timestamp': timestamp,
            'summary': analysis['summary']
        })
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/state', methods=['GET'])
def get_state():
    return jsonify({
        'latestPerCamera': latest_per_camera,
        'recentEvents': events[:50],
        'totalEvents': len(events)
    })

@app.route('/logs', methods=['GET'])
def view_logs():
    try:
        with open('flask_server.log', 'r') as f:
            logs = f.read()
        return f"<pre>{logs}</pre>", 200, {'Content-Type': 'text/html'}
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/')
def index():
    try:
        # Serve the HTML frontend
        with open('frontend.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Fallback status page
        html = f"""
        <html>
        <head><title>Advanced AI Detection Server</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h1>🤖 Advanced AI Detection Server</h1>
            <p><strong>Status:</strong> Running</p>
            <p style="color: red;"><strong>Error:</strong> frontend.html not found. Please create it in the same directory.</p>
            
            <h2>AI Models</h2>
            <ul>
                <li>YOLOv8: {'✅ Loaded' if yolo_model else '❌ Not loaded'}</li>
                <li>BLIP (Image Captioning): {'✅ Loaded' if blip_model else '❌ Not loaded'}</li>
            </ul>
            
            <h2>Stats</h2>
            <ul>
                <li>Total Requests: {request_count}</li>
                <li>Total Events: {len(events)}</li>
                <li>Cameras: {len(latest_per_camera)}</li>
            </ul>
            
            <h2>Links</h2>
            <ul>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/state">Current State</a></li>
                <li><a href="/logs">View Logs</a></li>
            </ul>
        </body>
        </html>
        """
        return html

@socketio.on('connect')
def handle_connect():
    logger.info("✅ Socket.IO client connected")
    emit('init', {
        'latestPerCamera': latest_per_camera,
        'recentEvents': events[:50]
    })

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("❌ Socket.IO client disconnected")

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("🤖 ADVANCED AI DETECTION SERVER")
    logger.info("="*60)
    logger.info(f"ESP32: http://10.118.99.215:5000")
    logger.info(f"Web: http://localhost:5000")
    logger.info("="*60)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)