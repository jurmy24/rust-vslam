use nalgebra::{Matrix3, UnitQuaternion, Vector3};

/// Gravity vector in world frame (m/s^2).
pub const GRAVITY: Vector3<f64> = Vector3::new(0.0, 0.0, -9.81);

/// IMU noise parameters (1-sigma).
#[derive(Debug, Clone, Copy)]
pub struct ImuNoise {
    pub sigma_gyro: f64,
    pub sigma_accel: f64,
}

impl ImuNoise {
    pub fn default() -> Self {
        Self {
            // Approximate EuRoC noise values
            sigma_gyro: 1.7e-4,
            sigma_accel: 2.0e-3,
        }
    }
}

/// IMU biases.
#[derive(Debug, Clone, Copy)]
pub struct ImuBias {
    pub gyro: Vector3<f64>,
    pub accel: Vector3<f64>,
}

impl ImuBias {
    pub fn zero() -> Self {
        Self {
            gyro: Vector3::zeros(),
            accel: Vector3::zeros(),
        }
    }
}

/// Single IMU measurement.
#[derive(Debug, Clone, Copy)]
pub struct ImuSample {
    pub timestamp_s: f64,
    pub accel: Vector3<f64>,
    pub gyro: Vector3<f64>,
}

/// Preintegrated motion between two frames.
#[derive(Debug, Clone, Copy)]
pub struct PreintegratedState {
    pub delta_rot: UnitQuaternion<f64>,
    pub delta_vel: Vector3<f64>,
    pub delta_pos: Vector3<f64>,
    pub dt: f64,
}

impl PreintegratedState {
    pub fn identity() -> Self {
        Self {
            delta_rot: UnitQuaternion::identity(),
            delta_vel: Vector3::zeros(),
            delta_pos: Vector3::zeros(),
            dt: 0.0,
        }
    }
}

/// IMU preintegrator: integrates high-rate IMU into a relative motion prior.
pub struct Preintegrator {
    pub bias: ImuBias,
    pub noise: ImuNoise,
    pub state: PreintegratedState,
}

impl Preintegrator {
    pub fn new(bias: ImuBias, noise: ImuNoise) -> Self {
        Self {
            bias,
            noise,
            state: PreintegratedState::identity(),
        }
    }

    pub fn reset(&mut self) {
        self.state = PreintegratedState::identity();
    }

    /// Integrate a single time step using midpoint integration.
    pub fn integrate(&mut self, prev: ImuSample, curr: ImuSample) {
        let dt = curr.timestamp_s - prev.timestamp_s;
        if dt <= 0.0 {
            return;
        }

        let gyro_prev = prev.gyro - self.bias.gyro;
        let gyro_curr = curr.gyro - self.bias.gyro;
        let omega = 0.5 * (gyro_prev + gyro_curr);

        // Update rotation
        let angle_axis = omega * dt;
        let delta_q = UnitQuaternion::from_scaled_axis(angle_axis);
        self.state.delta_rot = self.state.delta_rot * delta_q;

        // Rotate acceleration into world frame (approx using updated rotation)
        let accel_prev = prev.accel - self.bias.accel;
        let accel_curr = curr.accel - self.bias.accel;
        let accel_body = 0.5 * (accel_prev + accel_curr);
        let accel_world = self.state.delta_rot * accel_body + GRAVITY;

        // Update velocity and position
        self.state.delta_vel += accel_world * dt;
        self.state.delta_pos += self.state.delta_vel * dt + 0.5 * accel_world * dt * dt;
        self.state.dt += dt;
    }

    /// Predict pose/velocity from a previous state.
    pub fn propagate(
        &self,
        prev_rot: UnitQuaternion<f64>,
        prev_pos: Vector3<f64>,
        prev_vel: Vector3<f64>,
    ) -> (UnitQuaternion<f64>, Vector3<f64>, Vector3<f64>) {
        let rot = prev_rot * self.state.delta_rot;
        let vel = prev_vel + self.state.delta_vel;
        let pos = prev_pos + self.state.delta_pos + prev_vel * self.state.dt;
        (rot, pos, vel)
    }

    /// Convenience: build rotation matrix of current delta.
    pub fn delta_rotation_matrix(&self) -> Matrix3<f64> {
        self.state.delta_rot.to_rotation_matrix().into_inner()
    }
}
