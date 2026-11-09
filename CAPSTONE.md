# Nighttime Vision Sensor Attenuation in Dense Urban Platoon Corridors

**Research question.** When does degraded camera information become unsafe for automated emergency braking? The hypothesis is that poor illumination and sensor noise lower camera confidence, increasing the chance of an adverse event during tightly spaced platoon operation.

### Cohort and outcome

The exact stress cohort requires both Dark-No-Lights lighting and the Camera_Blind disengagement reason. The NHTSA incident reports provide lighting context. They lack structured camera-blind coding, synchronized sensor confidence, and a denominator of successful braking runs.

Y = 1 if emergency braking failure or unsafe takeover is recorded. Y = 0 only when both are known absent. The label stays missing otherwise. The classifier uses pre-action measurements. Takeover latency is analyzed separately to avoid including an outcome-dependent predictor.

### Models and evaluation

- Camera attenuation is fitted on a separate lit/unlit comparison cohort: confidence = β₀ + β₁L + β₂N + β₃LN, where L is illumination in units of 10 lux and N is normalized noise. The held-out error describes prediction, without establishing a causal effect.
- The adverse-outcome model is regularized logistic regression using two PCA scores from standardized camera, radar, and lidar confidence plus standardized light, noise, speed and gap. Compare it with the training prevalence and a camera-only logistic regression.
- A grouped split keeps each session entirely within training, calibration, or final evaluation. The minimum and maximum for latency scaling come from training. Missing values remain missing.
- The camera threshold is a separate decision rule. Screen calibration candidates using retained support and a one-sided Wilson upper bound. On held-out sessions, report adverse counts, rate intervals from resampling sessions, and the false-negative rate among adverse cases. The screening bound assumes independent rows and does not account for searching multiple thresholds. A candidate is not a certification of safety.

### STPA and simulator

| Unsafe action | Cause | Simulator check |
|---|---|---|
| Omit braking for an obstacle | Camera fails to detect the obstacle | Low camera with obstacle |
| Brake unnecessarily | False detection accepted | No obstacle |
| Brake too late | Slow actuator feedback | Late actuator feedback |
| Release too soon | Gap estimate recovers while the obstacle remains | Obstacle still present after request |

The governor uses normal fusion with a usable camera above or at the gate. Otherwise it transfers braking authority to usable radar, enters reduced automation if only lidar remains usable, or requests a simulated minimum-risk maneuver when all channels fail. A detected obstacle within the illustrative TTC gate, or an obstacle that persists after a previous request, triggers braking in an available authority mode.

### Limits

A field evaluation needs synchronized, representative telemetry covering successful and adverse events, held-out vehicles and routes, and a review of exposure and sampling design. Uploaded observations support sample-level comparisons. The simulator does not interface with an actuator.

Assignment source: CAPSTONE PROJECTS (1), Project 2. Worked-method reference: PROJECT 2. Incident source: [NHTSA Standing General Order crash reporting archive](https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting).
