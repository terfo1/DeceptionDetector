"""CSV writer for raw eye-tracking experiment data."""

import csv
from pathlib import Path


class CsvDataWriter:
    """Append experiment rows to the raw dataset CSV files."""

    PARTICIPANTS_HEADER = ["participant_id", "notes"]
    SESSIONS_HEADER = [
        "session_id",
        "participant_id",
        "date",
        "device",
        "screen_width",
        "screen_height",
        "sampling_rate",
        "calibration_quality",
    ]
    TRIALS_HEADER = [
        "trial_id",
        "session_id",
        "question_text",
        "instruction",
        "label",
        "answer",
        "response_time",
        "start_time",
        "end_time",
    ]
    GAZE_SAMPLES_HEADER = [
        "sample_id",
        "trial_id",
        "timestamp",
        "gaze_x",
        "gaze_y",
        "pupil_left",
        "pupil_right",
        "blink",
        "fixation",
        "saccade",
        "validity",
    ]
    SESSION_METADATA_HEADER = [
        "session_id",
        "participant_id",
        "experiment_version",
        "protocol_version",
        "operator_notes",
        "calibration_status",
        "baseline_duration_seconds",
        "device",
        "screen_width",
        "screen_height",
        "sampling_rate",
        "created_at",
    ]

    def __init__(self, raw_dir=None):
        if raw_dir is None:
            project_root = Path(__file__).resolve().parents[2]
            raw_dir = project_root / "data" / "raw"

        self.raw_dir = Path(raw_dir)
        self.participants_path = self.raw_dir / "participants.csv"
        self.sessions_path = self.raw_dir / "sessions.csv"
        self.trials_path = self.raw_dir / "trials.csv"
        self.gaze_samples_path = self.raw_dir / "gaze_samples.csv"
        self.session_metadata_path = self.raw_dir / "session_metadata.csv"

    def ensure_raw_files(self):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_file(self.participants_path, self.PARTICIPANTS_HEADER)
        self._ensure_file(self.sessions_path, self.SESSIONS_HEADER)
        self._ensure_file(self.trials_path, self.TRIALS_HEADER)
        self._ensure_file(self.gaze_samples_path, self.GAZE_SAMPLES_HEADER)
        self._ensure_file(self.session_metadata_path, self.SESSION_METADATA_HEADER)
        from .participant_metadata import ensure_participant_metadata_file

        ensure_participant_metadata_file(self.raw_dir / "participant_metadata.csv")

    def append_participant(self, participant_id, notes):
        self._append_row(
            self.participants_path,
            self.PARTICIPANTS_HEADER,
            {"participant_id": participant_id, "notes": notes},
        )

    def append_session(
        self,
        session_id,
        participant_id,
        date,
        device,
        screen_width,
        screen_height,
        sampling_rate,
        calibration_quality,
    ):
        self._append_row(
            self.sessions_path,
            self.SESSIONS_HEADER,
            {
                "session_id": session_id,
                "participant_id": participant_id,
                "date": date,
                "device": device,
                "screen_width": screen_width,
                "screen_height": screen_height,
                "sampling_rate": sampling_rate,
                "calibration_quality": calibration_quality,
            },
        )

    def append_trial(
        self,
        trial_id,
        session_id,
        question_text,
        instruction,
        label,
        answer,
        response_time,
        start_time,
        end_time,
    ):
        self._append_row(
            self.trials_path,
            self.TRIALS_HEADER,
            {
                "trial_id": trial_id,
                "session_id": session_id,
                "question_text": question_text,
                "instruction": instruction,
                "label": label,
                "answer": answer,
                "response_time": response_time,
                "start_time": start_time,
                "end_time": end_time,
            },
        )

    def append_gaze_samples(self, samples):
        if not samples:
            return

        self.ensure_raw_files()
        with self.gaze_samples_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.GAZE_SAMPLES_HEADER)
            for sample in samples:
                writer.writerow(sample)

    def append_participant_metadata(self, row):
        from .participant_metadata import append_participant_metadata

        return append_participant_metadata(row, self.raw_dir / "participant_metadata.csv")

    def append_session_metadata(
        self,
        session_id,
        participant_id,
        experiment_version,
        protocol_version,
        operator_notes,
        calibration_status,
        baseline_duration_seconds,
        device,
        screen_width,
        screen_height,
        sampling_rate,
        created_at,
    ):
        self._append_row(
            self.session_metadata_path,
            self.SESSION_METADATA_HEADER,
            {
                "session_id": session_id,
                "participant_id": participant_id,
                "experiment_version": experiment_version,
                "protocol_version": protocol_version,
                "operator_notes": operator_notes,
                "calibration_status": calibration_status,
                "baseline_duration_seconds": baseline_duration_seconds,
                "device": device,
                "screen_width": screen_width,
                "screen_height": screen_height,
                "sampling_rate": sampling_rate,
                "created_at": created_at,
            },
        )

    def append_session_quality(self, row):
        from .session_quality import append_session_quality

        append_session_quality(row, self.raw_dir / "session_quality.csv")

    def get_next_session_id(self):
        return self._next_prefixed_id(self.sessions_path, "session_id", "S")

    def get_next_trial_id(self):
        return self._next_prefixed_id(self.trials_path, "trial_id", "T")

    def get_next_sample_id(self):
        self.ensure_raw_files()
        max_id = 0
        with self.gaze_samples_path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                try:
                    max_id = max(max_id, int(row["sample_id"]))
                except (KeyError, TypeError, ValueError):
                    continue
        return max_id + 1

    def _ensure_file(self, path, header):
        if not path.exists() or path.stat().st_size == 0:
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(header)

    def _append_row(self, path, header, row):
        self.ensure_raw_files()
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writerow(row)

    def _next_prefixed_id(self, path, column_name, prefix):
        self.ensure_raw_files()
        max_number = 0
        with path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                value = row.get(column_name, "")
                if not value.startswith(prefix):
                    continue
                try:
                    max_number = max(max_number, int(value[len(prefix) :]))
                except ValueError:
                    continue
        return f"{prefix}{max_number + 1:03d}"
