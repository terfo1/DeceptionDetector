"""Built-in statement-level trials for the deception experiment."""


def get_statement_level_trials():
    """Return the default statement-level truth/lie trial list."""
    questions = [
        "Are you currently looking at a screen?",
        "Are you sitting on a chair?",
        "Is this experiment running on a computer?",
        "Are you answering this question now?",
    ]

    trials = []
    for question in questions:
        trials.append(
            {
                "question_text": question,
                "instruction": "truth",
                "label": 0,
            }
        )

    for question in questions:
        trials.append(
            {
                "question_text": question,
                "instruction": "lie",
                "label": 1,
            }
        )

    return trials
