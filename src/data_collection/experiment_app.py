"""Tkinter experiment runner for collecting labeled trial data."""

import random
import time
import tkinter as tk
from datetime import date
from tkinter import messagebox

from .data_writer import CsvDataWriter
from .mock_eye_tracker import MockEyeTracker
from .question_bank import get_statement_level_trials


class ExperimentApp:
    """Simple local interface for the statement-level deception task."""

    FIXATION_SECONDS = 2
    INSTRUCTION_SECONDS = 1
    QUESTION_TIMEOUT_SECONDS = 5
    PAUSE_SECONDS = 1
    BASELINE_SECONDS = 5

    def __init__(self, root):
        self.root = root
        self.root.title("Eye-Tracking Deception Risk Data Collection")
        self.root.geometry("900x600")
        self.root.configure(bg="white")

        self.writer = CsvDataWriter()
        self.eye_tracker = MockEyeTracker(sampling_rate=60)

        self.participant_id = None
        self.notes = ""
        self.session_id = None
        self.session_start_time = None
        self.trials = []
        self.current_trial_index = 0
        self.current_trial = None
        self.current_trial_id = None
        self.current_trial_start_time = None
        self.question_start_perf = None
        self.question_timeout_job = None
        self.completed_trials = 0

        try:
            self.writer.ensure_raw_files()
        except OSError as error:
            messagebox.showerror("Data error", f"Could not prepare raw CSV files:\n{error}")

        self.show_start_screen()

    def show_start_screen(self):
        self._clear()
        self._center_label(
            "Eye-Tracking Deception Risk Data Collection",
            size=24,
            pady=(80, 20),
        )
        self._center_label(
            "This tool collects labeled eye-tracking trial data for a controlled deception experiment.",
            size=18,
            pady=(0, 40),
        )
        self._button("Start", self.show_participant_screen).pack(pady=20)

    def show_participant_screen(self):
        self._clear()
        self._center_label("Participant setup", size=24, pady=(70, 30))

        form = tk.Frame(self.root, bg="white")
        form.pack()

        tk.Label(form, text="Participant ID", font=("Arial", 16), bg="white").grid(
            row=0, column=0, sticky="e", padx=10, pady=10
        )
        self.participant_entry = tk.Entry(form, font=("Arial", 16), width=24)
        self.participant_entry.insert(0, "P001")
        self.participant_entry.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(form, text="Notes", font=("Arial", 16), bg="white").grid(
            row=1, column=0, sticky="e", padx=10, pady=10
        )
        self.notes_entry = tk.Entry(form, font=("Arial", 16), width=24)
        self.notes_entry.grid(row=1, column=1, padx=10, pady=10)

        self._button("Continue", self.start_session).pack(pady=30)

    def start_session(self):
        participant_id = self.participant_entry.get().strip()
        notes = self.notes_entry.get().strip()

        if not participant_id:
            messagebox.showwarning("Missing participant ID", "Participant ID is required.")
            return

        self.participant_id = participant_id
        self.notes = notes
        self.session_id = self.writer.get_next_session_id()
        self.session_start_time = time.perf_counter()

        try:
            self.writer.append_participant(self.participant_id, self.notes)
            self.writer.append_session(
                session_id=self.session_id,
                participant_id=self.participant_id,
                date=date.today().isoformat(),
                device="MockEyeTracker",
                screen_width=self.root.winfo_screenwidth(),
                screen_height=self.root.winfo_screenheight(),
                sampling_rate=60,
                calibration_quality="mock_good",
            )
        except OSError as error:
            messagebox.showerror("Data error", f"Could not save participant/session data:\n{error}")
            return

        self.trials = get_statement_level_trials()
        random.shuffle(self.trials)
        self.show_calibration_screen()

    def show_calibration_screen(self):
        self._clear()
        self._center_label("Calibration placeholder", size=24, pady=(120, 20))
        self._center_label(
            "Real eye-tracker calibration will be added later.",
            size=18,
            pady=(0, 40),
        )
        self._button("Continue", self.show_baseline_screen).pack(pady=20)

    def show_baseline_screen(self):
        self.baseline_remaining = self.BASELINE_SECONDS
        self._show_baseline_countdown()

    def _show_baseline_countdown(self):
        self._clear()
        self._center_label("Baseline recording placeholder", size=24, pady=(100, 20))
        self._center_label("Participant should look at the fixation point.", size=18, pady=(0, 30))
        self._center_label("+", size=42, pady=(0, 20))
        self._center_label(str(self.baseline_remaining), size=24)

        if self.baseline_remaining <= 0:
            self.root.after(500, self.start_trial_loop)
            return

        self.baseline_remaining -= 1
        self.root.after(1000, self._show_baseline_countdown)

    def start_trial_loop(self):
        self.current_trial_index = 0
        self.completed_trials = 0
        self.show_next_trial()

    def show_next_trial(self):
        if self.current_trial_index >= len(self.trials):
            self.show_end_screen()
            return

        self.current_trial = self.trials[self.current_trial_index]
        self.current_trial_id = self.writer.get_next_trial_id()
        self.current_trial_start_time = self._session_elapsed()
        self.show_fixation()

    def show_fixation(self):
        self._clear()
        self._center_label("+", size=48, pady=(200, 0))
        self.root.after(self.FIXATION_SECONDS * 1000, self.show_instruction)

    def show_instruction(self):
        self._clear()
        instruction = self.current_trial["instruction"]
        if instruction == "truth":
            text = "Instruction: Answer truthfully"
        else:
            text = "Instruction: Answer deceptively"
        self._center_label(text, size=24, pady=(200, 0))
        self.root.after(self.INSTRUCTION_SECONDS * 1000, self.show_question)

    def show_question(self):
        self._clear()
        self.question_start_perf = time.perf_counter()
        self.eye_tracker.start_trial(self.current_trial_id)

        self._center_label(self.current_trial["question_text"], size=22, pady=(120, 40))

        buttons = tk.Frame(self.root, bg="white")
        buttons.pack()
        self._button("Yes", lambda: self.handle_answer("yes"), parent=buttons).grid(
            row=0, column=0, padx=20
        )
        self._button("No", lambda: self.handle_answer("no"), parent=buttons).grid(
            row=0, column=1, padx=20
        )

        self.question_timeout_job = self.root.after(
            self.QUESTION_TIMEOUT_SECONDS * 1000,
            self.handle_timeout,
        )

    def handle_answer(self, answer):
        if self.question_timeout_job is not None:
            self.root.after_cancel(self.question_timeout_job)
            self.question_timeout_job = None

        response_time = min(
            time.perf_counter() - self.question_start_perf,
            self.QUESTION_TIMEOUT_SECONDS,
        )
        self.save_current_trial(answer=answer, response_time=response_time)

    def handle_timeout(self):
        self.question_timeout_job = None
        self.save_current_trial(answer="timeout", response_time=self.QUESTION_TIMEOUT_SECONDS)

    def save_current_trial(self, answer, response_time):
        trial_end_time = self._session_elapsed()
        response_time_text = f"{response_time:.2f}"
        start_time_text = f"{self.current_trial_start_time:.2f}"
        end_time_text = f"{trial_end_time:.2f}"
        instruction = self.current_trial["instruction"]
        label = 0 if instruction == "truth" else 1

        try:
            next_sample_id = self.writer.get_next_sample_id()
            samples = self.eye_tracker.generate_samples(response_time, next_sample_id)
            self.eye_tracker.stop_trial()

            self.writer.append_trial(
                trial_id=self.current_trial_id,
                session_id=self.session_id,
                question_text=self.current_trial["question_text"],
                instruction=instruction,
                label=label,
                answer=answer,
                response_time=response_time_text,
                start_time=start_time_text,
                end_time=end_time_text,
            )
            self.writer.append_gaze_samples(samples)
        except (OSError, ValueError) as error:
            messagebox.showerror("Data error", f"Could not save trial data:\n{error}")
            self.show_end_screen()
            return

        self.completed_trials += 1
        self.current_trial_index += 1
        self.show_pause()

    def show_pause(self):
        self._clear()
        self._center_label("Please wait", size=24, pady=(220, 0))
        self.root.after(self.PAUSE_SECONDS * 1000, self.show_next_trial)

    def show_end_screen(self):
        self._clear()
        self._center_label("Experiment completed", size=24, pady=(150, 20))
        self._center_label("Data saved to data/raw", size=18, pady=(0, 40))
        self._button("Close", self.root.destroy).pack(pady=20)

        print("Experiment completed")
        print(f"participant_id: {self.participant_id}")
        print(f"session_id: {self.session_id}")
        print(f"trials_completed: {self.completed_trials}")
        print(f"raw_data_path: {self.writer.raw_dir}")

    def _session_elapsed(self):
        return time.perf_counter() - self.session_start_time

    def _clear(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def _center_label(self, text, size=18, pady=0):
        label = tk.Label(
            self.root,
            text=text,
            font=("Arial", size),
            bg="white",
            fg="black",
            wraplength=760,
            justify="center",
        )
        label.pack(pady=pady)
        return label

    def _button(self, text, command, parent=None):
        if parent is None:
            parent = self.root
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Arial", 16),
            width=12,
            height=2,
        )


def main():
    root = tk.Tk()
    ExperimentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
