use anyhow::Result;
use nalgebra::Vector3;
use opencv::core::{DMatch, Mat, Vector};
use opencv::features2d::BFMatcher;
use opencv::prelude::*;

use crate::frontend::camera::CameraModel;
use crate::frontend::features::{FeatureDetector, FeatureSet, keypoints_to_points};

#[derive(Clone)]
pub struct StereoFrame {
    pub left_features: FeatureSet,
    pub right_features: FeatureSet,
    pub matches_lr: Vector<DMatch>,
    /// 3D points per left keypoint index (None if no valid depth).
    pub points_cam: Vec<Option<Vector3<f64>>>,
    pub timestamp_ns: u64,
}

pub struct StereoProcessor {
    detector: FeatureDetector,
    matcher: BFMatcher,
    camera: CameraModel,
}

impl StereoProcessor {
    pub fn new(camera: CameraModel, n_features: i32) -> Result<Self> {
        let detector = FeatureDetector::new(n_features)?;
        let matcher = BFMatcher::new(opencv::core::NORM_HAMMING, true)?;
        Ok(Self {
            detector,
            matcher,
            camera,
        })
    }

    pub fn process(&mut self, left: &Mat, right: &Mat, timestamp_ns: u64) -> Result<StereoFrame> {
        let left_features = self.detector.detect(left)?;
        let right_features = self.detector.detect(right)?;

        let matches_lr = self.match_features(&left_features, &right_features)?;
        let points_cam = triangulate(&left_features, &right_features, &matches_lr, self.camera);

        Ok(StereoFrame {
            left_features,
            right_features,
            matches_lr,
            points_cam,
            timestamp_ns,
        })
    }

    fn match_features(&self, left: &FeatureSet, right: &FeatureSet) -> Result<Vector<DMatch>> {
        let mut matches = Vector::<DMatch>::new();
        self.matcher.train_match(
            &left.descriptors,
            &right.descriptors,
            &mut matches,
            &Mat::default(),
        )?;
        Ok(matches)
    }
}

/// Triangulate 3D points from stereo matches using pinhole model.
fn triangulate(
    left: &FeatureSet,
    right: &FeatureSet,
    matches: &Vector<DMatch>,
    cam: CameraModel,
) -> Vec<Option<Vector3<f64>>> {
    let left_pts = keypoints_to_points(&left.keypoints);
    let right_pts = keypoints_to_points(&right.keypoints);

    let mut points = vec![None; left_pts.len()];

    for m in matches {
        if let (Some(l), Some(r)) = (
            left_pts.get(m.query_idx as usize),
            right_pts.get(m.train_idx as usize),
        ) {
            let disparity = l.x - r.x;
            if disparity.abs() < 0.5 {
                continue;
            }
            let z = cam.fx * cam.baseline / disparity;
            let x = (l.x - cam.cx) * z / cam.fx;
            let y = (l.y - cam.cy) * z / cam.fy;
            points[m.query_idx as usize] = Some(Vector3::new(x, y, z));
        }
    }

    points
}
