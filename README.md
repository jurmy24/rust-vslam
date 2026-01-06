# rust-vslam

A Visual SLAM (Simultaneous Localization and Mapping) implementation in Rust using OpenCV. This project implements computer vision algorithms for tracking camera position and building 3D maps from camera feeds.

## Features

- Real-time camera feed capture and processing
- ORB (Oriented FAST and Rotated BRIEF) feature detection
- Keypoint visualization and tracking
- Foundation for 3D pose estimation and mapping

## Prerequisites

- Rust (edition 2024)
- OpenCV 4.x
- LLVM/Clang (for OpenCV bindings)
- A webcam or camera device

### macOS Setup

Install dependencies via Homebrew:

```bash
brew install llvm opencv
```

Set environment variables to help the compiler locate libraries:

```bash
export LIBCLANG_PATH="$(brew --prefix llvm)/lib"
export DYLD_LIBRARY_PATH="$(brew --prefix llvm)/lib:$DYLD_LIBRARY_PATH"
export PATH="$(brew --prefix llvm)/bin:$PATH"
```

## Building

```bash
# Debug build
cargo build

# Release build (recommended for real-time performance)
cargo build --release
```

## Running

The project includes multiple binaries for different VSLAM components:

```bash
# Camera feed test
cargo run --bin start-camera

# Feature detection and landmark tracking
cargo run --bin feature-landmarks

# Main entry point (placeholder)
cargo run
```

## Project Structure

```
src/
├── main.rs                      # Main entry point
└── bin/
    ├── start-camera.rs          # Camera initialization and feed display
    └── feature-landmarks.rs     # ORB feature detection and visualization
```

## Architecture

This implementation focuses on the "front-end" of a VSLAM system:

1. **Camera Input**: Captures frames from webcam
2. **Preprocessing**: Converts BGR to grayscale
3. **Feature Detection**: Uses ORB detector (up to 1000 keypoints)
4. **Visualization**: Real-time keypoint overlay

Future additions will include:
- Camera pose estimation (using nalgebra)
- 3D map visualization (using rerun)
- Bundle adjustment and loop closure

## Dependencies

- **opencv** (0.98.1) - Computer vision operations and feature detection
- **nalgebra** (0.34.1) - Linear algebra for 3D transformations
- **rerun** (0.28.1) - Visualization toolkit
- **anyhow** (1.0.100) - Error handling

## License

This project is open source.
