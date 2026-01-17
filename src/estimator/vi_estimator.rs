use std::collections::HashMap;

use anyhow::Result;
use nalgebra::{Matrix3, Vector3};
use opencv::calib3d;
use opencv::core::{Mat, Point2f, Point3d};
use opencv::prelude::*;

use crate::frontend::camera::CameraModel;
use crate::frontend::stereo::StereoFrame;
use crate::frontend::tracker::{TemporalMatchResult, TemporalMatcher};
use crate::imu::{ImuBias, ImuNoise, Preintegrator};
use crate::io::euroc::ImuEntry;
use crate::math::SE3;

pub struct FrameState {
    pub frame: StereoFrame,
    /// Map from left keypoint index to world point.
    pub world_points: HashMap<i32, Vector3<f64>>,
}

pub struct VisualInertialEstimator {
    camera: CameraModel,
    temporal: TemporalMatcher,
    preintegrator: Preintegrator,
    pub pose: SE3,
    pub velocity: Vector3<f64>,
    pub trajectory: Vec<SE3>,
    prev_state: Option<FrameState>,
}

impl VisualInertialEstimator {
    pub fn new(camera: CameraModel) -> Result<Self> {
        Ok(Self {
            camera,
            temporal: TemporalMatcher::new()?,
            preintegrator: Preintegrator::new(ImuBias::zero(), ImuNoise::default()),
            pose: SE3::identity(),
            velocity: Vector3::zeros(),
            trajectory: vec![SE3::identity()],
            prev_state: None,
        })
    }

    /// Process a stereo frame with accompanying IMU interval.
    pub fn process_frame(
        &mut self,
        stereo_frame: StereoFrame,
        imu_measurements: &[ImuEntry],
    ) -> Result<SE3> {
        // Integrate IMU for motion prior
        self.preintegrator.reset();
        for pair in imu_measurements.windows(2) {
            let prev = pair[0].sample;
            let curr = pair[1].sample;
            self.preintegrator.integrate(prev, curr);
        }

        let (pred_rot, pred_pos, pred_vel) =
            self.preintegrator
                .propagate(self.pose.rotation, self.pose.translation, self.velocity);
        let imu_prior = SE3 {
            rotation: pred_rot,
            translation: pred_pos,
        };

        let estimated_pose = if let Some(prev_state) = &self.prev_state {
            let matches = self
                .temporal
                .match_features(&prev_state.frame.left_features, &stereo_frame.left_features)?;
            let (points3d, points2d) =
                self.build_correspondences(prev_state, &stereo_frame, &matches);

            if points3d.is_empty() {
                imu_prior
            } else {
                self.solve_pnp(&points3d, &points2d, Some(&imu_prior))
                    .unwrap_or(imu_prior)
            }
        } else {
            imu_prior
        };

        // Update state
        self.pose = estimated_pose;
        self.velocity = pred_vel;
        self.trajectory.push(self.pose.clone());

        let world_points = self.build_world_points(&stereo_frame, &self.pose);
        self.prev_state = Some(FrameState {
            frame: stereo_frame,
            world_points,
        });

        Ok(self.pose.clone())
    }

    fn build_correspondences(
        &self,
        prev: &FrameState,
        curr: &StereoFrame,
        matches: &TemporalMatchResult,
    ) -> (Vec<Vector3<f64>>, Vec<Point2f>) {
        let mut pts3d = Vec::new();
        let mut pts2d = Vec::new();

        for m in matches.matches.iter() {
            if let Some(world_pt) = prev.world_points.get(&m.query_idx) {
                if let Ok(kp) = curr.left_features.keypoints.get(m.train_idx as usize) {
                    pts3d.push(*world_pt);
                    pts2d.push(Point2f::new(kp.pt().x, kp.pt().y));
                }
            }
        }

        (pts3d, pts2d)
    }

    fn build_world_points(&self, frame: &StereoFrame, pose: &SE3) -> HashMap<i32, Vector3<f64>> {
        let mut map = HashMap::new();
        for (idx, p_opt) in frame.points_cam.iter().enumerate() {
            if let Some(p_cam) = p_opt {
                let p_world = pose.transform_point(p_cam);
                map.insert(idx as i32, p_world);
            }
        }
        map
    }

    fn solve_pnp(
        &self,
        points3d: &[Vector3<f64>],
        points2d: &[Point2f],
        prior: Option<&SE3>,
    ) -> Result<SE3> {
        // Convert to Point3d for opencv
        let pts3d: Vec<Point3d> = points3d
            .iter()
            .map(|p| Point3d::new(p.x, p.y, p.z))
            .collect();
        let obj_points = Mat::from_slice(&pts3d)?.try_clone()?;
        let img_points = Mat::from_slice(points2d)?.try_clone()?;

        let camera_matrix = Mat::from_slice_2d(&[
            [self.camera.fx, 0.0, self.camera.cx],
            [0.0, self.camera.fy, self.camera.cy],
            [0.0, 0.0, 1.0],
        ])?
        .try_clone()?;
        let dist_coeffs = Mat::zeros(1, 5, opencv::core::CV_64F)?.to_mat()?;

        let mut rvec = Mat::default();
        let mut tvec = Mat::default();
        let mut use_extrinsic_guess = false;

        if let Some(prior_pose) = prior {
            let rot_mat = prior_pose.rotation.to_rotation_matrix().into_inner();
            rvec = rotation_matrix_to_rvec(rot_mat)?;
            tvec = Mat::from_slice(&[
                prior_pose.translation.x,
                prior_pose.translation.y,
                prior_pose.translation.z,
            ])?
            .try_clone()?;
            use_extrinsic_guess = !rvec.empty();
        }

        calib3d::solve_pnp_ransac(
            &obj_points,
            &img_points,
            &camera_matrix,
            &dist_coeffs,
            &mut rvec,
            &mut tvec,
            use_extrinsic_guess,
            100,
            8.0,
            0.99,
            &mut opencv::core::no_array(),
            calib3d::SOLVEPNP_ITERATIVE,
        )?;

        let mut rot_mat = Mat::default();
        calib3d::rodrigues(&rvec, &mut rot_mat, &mut opencv::core::no_array())?;
        let rotation = mat3_to_matrix3(&rot_mat)?;
        let translation = Vector3::new(
            *tvec.at::<f64>(0i32)?,
            *tvec.at::<f64>(1i32)?,
            *tvec.at::<f64>(2i32)?,
        );

        Ok(SE3::from_rt(rotation, translation))
    }
}

fn rotation_matrix_to_rvec(rot: Matrix3<f64>) -> Result<Mat> {
    let rot_mat = Mat::from_slice(rot.as_slice())?.try_clone()?;
    let rot_mat_reshaped = rot_mat.reshape(1, 3)?.try_clone()?;
    let mut rvec = Mat::default();
    calib3d::rodrigues(&rot_mat_reshaped, &mut rvec, &mut opencv::core::no_array())?;
    Ok(rvec)
}

fn mat3_to_matrix3(mat: &Mat) -> Result<Matrix3<f64>> {
    let mut arr = [0.0f64; 9];
    for i in 0..9 {
        arr[i] = *mat.at::<f64>(i as i32)?;
    }
    Ok(Matrix3::from_row_slice(&arr))
}
