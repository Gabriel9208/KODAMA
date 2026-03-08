# KODAMA - Product Requirements Document (PRD)

## 1. Product Vision and Goals

### 1.1 Vision
Core Goal: Develop an "Edge Visual Understanding Engine" designed specifically for the visually impaired.

While autonomous driving has benefited from computer vision, egocentric scene understanding remains a challenge. Traditional white canes are limited by physical length and cannot detect suspended objects or fast-moving threats. Guide dogs are expensive and scarce.

This project aims to bridge this gap by creating a system that runs on wearable edge devices (performance $\le$ RTX 2060 equivalent) without cloud dependency, providing millisecond-level warnings for 3D dynamic obstacles in complex environments.

### 1.2 Goals         
**Product Goals:** 
1. **P0 (Core):** Real-time Panoptic Perception. Simultaneous processing of semantic and instance segmentation from monocular RGB images.
2. **P1 (Safety):** Zero-tolerance for close-range misses. Recall rate > 95% for obstacles within 2 meters.
3. **Constraint:** Inference delay < 50ms ( > 20 FPS) with VRAM usage < 4.5 GB to accommodate OS and drivers.

## 2. Functional Requirements

### 2.1 Feature (P0): Real-time Panoptic Perception & Separation
*   **User Story:** As a visually impaired person, I need the system to distinguish between immediate threats and safe walking areas so that I can navigate confidently.
*   **Acceptance Criteria:**
    *   **Things (Dynamic/Static Obstacles):** Accurately segment and bound objects like pedestrians, vehicles, poles, and furniture.
    *   **Stuff (Traversable Areas):** Clearly identify safe regions such as crosswalks, tactile paving, and flat sidewalks.
    *   System must resolve label conflicts when instance masks overlap with background semantic pixels.

### 2.2 Feature (P2): Extreme Environmental Adaptability
*   **User Story:** As a user, I want the system to be reliable even in poor weather or lighting conditions.
*   **Acceptance Criteria:**
    *   Maintain mask boundary smoothness and classification stability in low-light (dusk), strong noise, or reflective ground conditions (rain/puddles).
    *   Robustness against motion blur from body movement.

### 2.3 Feature (P1): Geometric Risk & Dynamic Warning
*   **User Story:** As a user, I need the system to prioritize warnings for objects that are close or approaching fast, as my safety margin is near zero.
*   **Acceptance Criteria:**
    *   **Recall Constraint:** For obstacles within 2 meters, the Recall rate must be > 95%.
    *   Distance estimation tolerance can be looser for objects > 10 meters away.
    *   Alert intensity must correlate with the urgency of the threat (distance + velocity).

### 2.4 Feature: Multi-modal Feedback Interface
*   **User Story:** As a user, I want to receive feedback via audio or haptic signals so that I can be aware of my surroundings without relying on sight.
*   **Acceptance Criteria:**
    *   Support for audio output (Text-to-Speech or warning tones) via headphones/speakers.
    *   Support for haptic feedback (vibration) if a wearable interface is connected.
    *   Users can customize the type of feedback (e.g., silent mode with vibration only).

## 3. Non-functional Requirements

### 3.1 Performance
*   **Response Time:** 
    1. Inference delay < 50ms per frame.
    2. System end-to-end latency (camera to feedback) less than 100ms.
*   **Frame Rate:** Minimum 15 FPS (Frames Per Second) to ensure smooth tracking.
*   **Resource Usage:** VRAM usage must be strictly < 4.5 GB during inference.

### 3.2 Hardware & Environment
*   **Compute Target:** Wearable/Portable edge devices with compute capability equivalent to or lower than NVIDIA RTX 2060.
*   **Connectivity:** Must function 100% offline (no cloud dependency).

### 3.3 Reliability
*   **Stability:** The system must run continuously for at least 1 hour without crashing or overheating.

## 4. Technical Architecture Strategy (CS231n Scope)

### 4.1 Neural Network Architecture
*   **Base Model:** Modified **YOLOv12-Nano**.
*   **Backbone:** **R-ELAN** architecture. Utilizes block-level residual connections and optimized feature aggregation to capture high-frequency edge details in shallow layers (crucial for mask quality).
*   **Neck & Head:** 
    *   Retain **Area Attention** for multi-scale feature processing.
    *   **Custom Segmentation Head:** Extended to output three tensor groups:
        1.  Bounding Box Coordinates.
        2.  Instance Segmentation Mask Coefficients.
        3.  Global Semantic Segmentation Feature Map.

### 4.2 Post-Processing
*   **Fusion Module:** A tensor-matrix operation based module.
*   **Logic:** Uses heuristic rules to resolve conflicts between Instance Masks (Things) and Background Semantics (Stuff).
