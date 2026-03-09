# KODAMA - Product Requirements Document (PRD)

## 1. Product Vision and Goals

### 1.1 Vision
Core Goal: Develop a robust "Egocentric Scene Understanding Engine" for uncontrollable environments.

While autonomous driving has benefited from computer vision, scene understanding from an egocentric (first-person) perspective remains a challenge due to motion blur, occlusion, and dynamic lighting.

This project aims to solve these challenges by creating a lightweight, real-time system capable of detecting 3D dynamic obstacles and traversable areas. The primary application scenario is assisting the visually impaired, serving as a technical verification for "Edge Visual Understanding".

### 1.2 Goals         
**Project Goals:** 
1. **P0 (Core):** Real-time Panoptic Perception. Simultaneous processing of semantic and instance segmentation from monocular RGB images.
2. **P1 (Safety):** Zero-tolerance for close-range misses. Recall rate > 95% for obstacles within 2 meters.
3. **Constraint:** Inference delay < 50ms ( > 20 FPS) with VRAM usage < 4.5 GB (Targeting mid-range consumer GPUs).

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
    *   System must output risk levels correlated with the urgency of the threat (distance + velocity).

## 3. Non-functional Requirements

### 3.1 Performance
*   **Response Time:** 
    1. Inference delay < 50ms per frame.
*   **Frame Rate:** Minimum 15 FPS (Frames Per Second) to ensure smooth tracking.
*   **Resource Usage:** VRAM usage must be strictly < 4.5 GB during inference.

## 4. Technical Architecture Strategy
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
