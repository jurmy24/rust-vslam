//! Rerun-based visualization for stereo SLAM.
//!
//! Entity hierarchy:
//!     camera/
//!         left/
//!             image       - Rectified left image
//!             features    - All detected features (green)
//!             matched     - Matched features (red)
//!         right/
//!             image       - Rectified right image
//!             features    - All detected features (green)
//!             matched     - Matched features (red)
//!     world/
//!         camera      - Camera pose transform
//!         trajectory  - Camera trajectory line
//!         points      - 3D point cloud (colored by depth)

use nalgebra::Vector3;
use opencv::core::{DMatch, KeyPoint, Mat, Vector};
use opencv::prelude::*;
use rerun::{RecordingStream, external::glam};

use crate::frontend::stereo::StereoFrame;
use crate::math::SE3;

pub struct RerunVisualizer {
    rec: RecordingStream,
}

impl RerunVisualizer {
    pub fn new(app_name: &str) -> Self {
        let rec = rerun::RecordingStreamBuilder::new(app_name)
            .spawn()
            .expect("Failed to spawn rerun viewer");

        // Set up coordinate system (Right-Down-Forward, typical camera convention)
        rec.log_static("world", &rerun::ViewCoordinates::RDF()).ok();

        Self { rec }
    }

    /// Set the current timestamp for all subsequent logs
    pub fn set_time(&self, timestamp_ns: u64) {
        let timestamp_sec = timestamp_ns as f64 / 1e9;
        self.rec.set_duration_secs("timestamp", timestamp_sec);
    }

    /// Log a complete stereo frame with images, features, and 3D points
    pub fn log_stereo_frame(&self, frame: &StereoFrame, left_image: &Mat, right_image: &Mat) {
        self.set_time(frame.timestamp_ns);
        self.log_images(left_image, right_image);
        self.log_features(
            &frame.left_features.keypoints,
            &frame.right_features.keypoints,
        );
        self.log_matches(
            &frame.left_features.keypoints,
            &frame.right_features.keypoints,
            &frame.matches_lr,
        );
        self.log_3d_points_from_frame(frame);
    }

    fn log_images(&self, left: &Mat, right: &Mat) {
        // Convert Mat to image data
        if let Ok((data, width, height)) = mat_to_image_data(left) {
            self.rec
                .log(
                    "camera/left/image",
                    &rerun::Image::from_l8(data, [width, height]),
                )
                .ok();
        }
        if let Ok((data, width, height)) = mat_to_image_data(right) {
            self.rec
                .log(
                    "camera/right/image",
                    &rerun::Image::from_l8(data, [width, height]),
                )
                .ok();
        }
    }

    fn log_features(&self, left_kps: &Vector<KeyPoint>, right_kps: &Vector<KeyPoint>) {
        // Left image features (green)
        let left_pts: Vec<[f32; 2]> = left_kps.iter().map(|kp| [kp.pt().x, kp.pt().y]).collect();
        if !left_pts.is_empty() {
            self.rec
                .log(
                    "camera/left/features",
                    &rerun::Points2D::new(left_pts)
                        .with_colors([[0u8, 255, 0]]) // Green
                        .with_radii([3.0f32]),
                )
                .ok();
        }

        // Right image features (green)
        let right_pts: Vec<[f32; 2]> = right_kps.iter().map(|kp| [kp.pt().x, kp.pt().y]).collect();
        if !right_pts.is_empty() {
            self.rec
                .log(
                    "camera/right/features",
                    &rerun::Points2D::new(right_pts)
                        .with_colors([[0u8, 255, 0]]) // Green
                        .with_radii([3.0f32]),
                )
                .ok();
        }
    }

    fn log_matches(
        &self,
        left_kps: &Vector<KeyPoint>,
        right_kps: &Vector<KeyPoint>,
        matches: &Vector<DMatch>,
    ) {
        if matches.is_empty() {
            return;
        }

        let mut left_matched: Vec<[f32; 2]> = Vec::new();
        let mut right_matched: Vec<[f32; 2]> = Vec::new();

        for m in matches.iter() {
            if let (Ok(lkp), Ok(rkp)) = (
                left_kps.get(m.query_idx as usize),
                right_kps.get(m.train_idx as usize),
            ) {
                left_matched.push([lkp.pt().x, lkp.pt().y]);
                right_matched.push([rkp.pt().x, rkp.pt().y]);
            }
        }

        // Matched points in left image (red)
        if !left_matched.is_empty() {
            self.rec
                .log(
                    "camera/left/matched",
                    &rerun::Points2D::new(left_matched)
                        .with_colors([[255u8, 0, 0]]) // Red
                        .with_radii([4.0f32]),
                )
                .ok();
        }

        // Matched points in right image (red)
        if !right_matched.is_empty() {
            self.rec
                .log(
                    "camera/right/matched",
                    &rerun::Points2D::new(right_matched)
                        .with_colors([[255u8, 0, 0]]) // Red
                        .with_radii([4.0f32]),
                )
                .ok();
        }
    }

    fn log_3d_points_from_frame(&self, frame: &StereoFrame) {
        // Collect valid 3D points
        let valid_points: Vec<Vector3<f64>> = frame
            .points_cam
            .iter()
            .filter_map(|p| *p)
            .filter(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
            .filter(|p| p.x.abs() < 100.0 && p.y.abs() < 100.0 && p.z > 0.1 && p.z < 100.0)
            .collect();

        if valid_points.is_empty() {
            return;
        }

        // Compute depth range for coloring
        let depths: Vec<f64> = valid_points.iter().map(|p| p.z).collect();
        let mut sorted_depths = depths.clone();
        sorted_depths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let depth_min = sorted_depths[sorted_depths.len() * 5 / 100]; // 5th percentile
        let depth_max = sorted_depths[sorted_depths.len() * 95 / 100]; // 95th percentile
        let depth_range = (depth_max - depth_min).max(0.1);

        // Convert to rerun format with depth-based colors
        let pts: Vec<[f32; 3]> = valid_points
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();

        // Color by depth: blue (close) -> green -> red (far)
        let colors: Vec<[u8; 3]> = depths
            .iter()
            .map(|&d| {
                let normalized = ((d - depth_min) / depth_range).clamp(0.0, 1.0);
                let r = (normalized * 255.0) as u8;
                let g = ((1.0 - (normalized - 0.5).abs() * 2.0) * 255.0) as u8;
                let b = ((1.0 - normalized) * 255.0) as u8;
                [r, g, b]
            })
            .collect();

        self.rec
            .log(
                "world/points",
                &rerun::Points3D::new(pts)
                    .with_colors(colors)
                    .with_radii([0.02f32]),
            )
            .ok();
    }

    pub fn log_pose(&self, pose: &SE3) {
        let translation = glam::Vec3::new(
            pose.translation.x as f32,
            pose.translation.y as f32,
            pose.translation.z as f32,
        );
        let rotation = glam::Quat::from_xyzw(
            pose.rotation.coords.x as f32,
            pose.rotation.coords.y as f32,
            pose.rotation.coords.z as f32,
            pose.rotation.w as f32,
        );
        self.rec
            .log(
                "world/camera",
                &rerun::Transform3D::from_translation_rotation(translation, rotation),
            )
            .ok();
    }

    pub fn log_trajectory(&self, positions: &[Vector3<f64>]) {
        if positions.len() < 2 {
            return;
        }
        let pts: Vec<[f32; 3]> = positions
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();

        // Yellow trajectory line
        self.rec
            .log(
                "world/trajectory",
                &rerun::LineStrips3D::new([pts.clone()])
                    .with_colors([[255u8, 255, 0]])
                    .with_radii([0.01f32]),
            )
            .ok();

        // Current position as cyan point
        if let Some(current) = pts.last() {
            self.rec
                .log(
                    "world/trajectory/current",
                    &rerun::Points3D::new([*current])
                        .with_colors([[0u8, 255, 255]]) // Cyan
                        .with_radii([0.05f32]),
                )
                .ok();
        }
    }

    pub fn log_map_points(&self, points: &[Vector3<f64>]) {
        if points.is_empty() {
            return;
        }
        let pts: Vec<[f32; 3]> = points
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        self.rec
            .log(
                "world/map",
                &rerun::Points3D::new(pts)
                    .with_colors([[0u8, 200, 255]])
                    .with_radii([0.03f32]),
            )
            .ok();
    }
}

/// Convert OpenCV Mat to image data (bytes, width, height)
fn mat_to_image_data(mat: &Mat) -> Result<(Vec<u8>, u32, u32), opencv::Error> {
    let rows = mat.rows() as u32;
    let cols = mat.cols() as u32;

    // Get raw data
    let data = mat.data_bytes()?;
    let image_data: Vec<u8> = data.to_vec();

    Ok((image_data, cols, rows))
}
