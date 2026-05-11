# Experiment Protocol

## Experiment type

The first version of the project uses a statement-level deception task.

Participants answer a set of simple questions. In some trials, they are instructed to answer truthfully. In other trials, they are instructed to answer deceptively. Eye-tracking data is recorded during each trial.

## Purpose

The purpose of this experiment is to collect labeled eye-tracking sequences for training and evaluating a deception risk detection model.

The system does not attempt to detect lies in uncontrolled real-world situations. It only analyzes gaze behavior during a controlled experimental task.

## Participant ID

Each participant receives an anonymous identifier:

- P001
- P002
- P003

Personal names are not stored in the dataset.

## Session structure

Each session follows this structure:

1. Participant registration
2. Eye-tracker calibration
3. Baseline recording
4. Training/example questions
5. Truth block
6. Lie block
7. Mixed block
8. End of experiment

## Calibration

Before the experiment starts, the participant completes eye-tracker calibration.

If calibration quality is poor, the session is repeated or excluded from the dataset.

## Baseline recording

A 60-second baseline is recorded before the main task.

During baseline recording, the participant looks at a neutral fixation point in the center of the screen.

The baseline is used to normalize pupil diameter and gaze behavior for each participant.

## Trial structure

Each trial consists of the following steps:

1. Fixation cross — 2 seconds
2. Instruction screen — 1 second
3. Question screen — maximum 5 seconds
4. Participant answer — yes/no
5. Pause — 1 second

## Blocks

### Block 1: Truth block

The participant answers all questions truthfully.

- Number of trials: 10
- Label: 0
- Meaning: truth

### Block 2: Lie block

The participant answers all questions deceptively.

- Number of trials: 10
- Label: 1
- Meaning: lie

### Block 3: Mixed block

Truth and lie trials are randomly mixed.

- Number of trials: 20
- Label 0: truth
- Label 1: lie

## Labels

The dataset uses binary labels:

- 0 = truthful response
- 1 = deceptive response

The label is based on the instruction given to the participant, not only on the yes/no answer.

## Recorded data

For each gaze sample, the following fields are recorded:

- participant_id
- session_id
- trial_id
- timestamp
- question_text
- instruction
- label
- answer
- response_time
- gaze_x
- gaze_y
- pupil_left
- pupil_right
- blink
- validity

Example:

participant_id,session_id,trial_id,timestamp,question_text,instruction,label,answer,response_time,gaze_x,gaze_y,pupil_left,pupil_right,blink,validity
P001,S001,T001,0.016,"Are you a student?","truth",0,"yes",1.42,0.51,0.43,3.21,3.18,0,1

## Minimum dataset size for prototype

The prototype dataset should include:

- 5–10 participants
- 40 trials per participant

This gives approximately 200–400 labeled trials.

## Important limitations

This protocol is designed for controlled experimental conditions.

The model trained on this data should not be interpreted as a universal lie detector. It should be interpreted as a model that estimates deception risk under this specific experimental protocol.