"""Mock eye tracker for generating structurally valid gaze samples."""

import random


class MockEyeTracker:
    """Generate fake gaze samples for local experiment testing."""

    def __init__(self, sampling_rate=60):
        self.sampling_rate = sampling_rate
        self.current_trial_id = None

    def start_trial(self, trial_id):
        self.current_trial_id = trial_id

    def stop_trial(self):
        self.current_trial_id = None

    def generate_samples(self, duration_seconds, start_sample_id):
        if self.current_trial_id is None:
            raise ValueError("No active trial. Call start_trial(trial_id) first.")

        duration_seconds = max(0.0, float(duration_seconds))
        sample_count = max(1, int(round(duration_seconds * self.sampling_rate)))
        interval = 1.0 / self.sampling_rate
        samples = []

        for index in range(sample_count):
            blink = 1 if random.random() < 0.03 else 0
            validity = 1 if random.random() < 0.97 else 0

            if blink:
                fixation = 0
                saccade = 0
                validity = 0
            else:
                fixation = 1 if random.random() < 0.75 else 0
                saccade = 0 if fixation else 1

            samples.append(
                {
                    "sample_id": start_sample_id + index,
                    "trial_id": self.current_trial_id,
                    "timestamp": f"{index * interval:.3f}",
                    "gaze_x": f"{self._clamp(random.gauss(0.5, 0.08)):.3f}",
                    "gaze_y": f"{self._clamp(random.gauss(0.5, 0.08)):.3f}",
                    "pupil_left": f"{random.gauss(3.3, 0.18):.2f}",
                    "pupil_right": f"{random.gauss(3.3, 0.18):.2f}",
                    "blink": blink,
                    "fixation": fixation,
                    "saccade": saccade,
                    "validity": validity,
                }
            )

        return samples

    @staticmethod
    def _clamp(value, lower=0.0, upper=1.0):
        return max(lower, min(upper, value))
