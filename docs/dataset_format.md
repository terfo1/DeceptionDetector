# Dataset Format

## Overview

The raw dataset is organized into four CSV files:
- participants.csv
- sessions.csv
- trials.csv
- gaze_samples.csv

This structure separates participant metadata, recording sessions, experimental trials, and raw eye-tracking samples. The files are connected by identifiers:
- participants.csv connects to sessions.csv through `participant_id`.
- sessions.csv connects to trials.csv through `session_id`.
- trials.csv connects to gaze_samples.csv through `trial_id`.

## Labels

The project uses binary labels:
- 0 = truthful response
- 1 = deceptive response

The label is based on the instruction shown to the participant, not only on the yes/no answer.

Example:
- Question: "Are you a student?"
- Real answer: yes
- Instruction: lie
- Participant answer: no
- Label: 1

## participants.csv

Columns:
- participant_id
- notes

Example:

```csv
participant_id,notes
P001,test participant
P002,test participant
```

`participant_id` is anonymous. Do not store real names.

## sessions.csv

Columns:
- session_id
- participant_id
- date
- device
- screen_width
- screen_height
- sampling_rate
- calibration_quality

Example:

```csv
session_id,participant_id,date,device,screen_width,screen_height,sampling_rate,calibration_quality
S001,P001,2026-05-07,Tobii,1920,1080,60,good
```

Fields:
- session_id: unique recording session
- participant_id: participant who completed the session
- date: recording date
- device: eye tracker or recording device
- screen_width: screen width in pixels
- screen_height: screen height in pixels
- sampling_rate: eye-tracking sampling rate in Hz
- calibration_quality: quality of eye-tracker calibration

## trials.csv

Columns:
- trial_id
- session_id
- question_text
- instruction
- label
- answer
- response_time
- start_time
- end_time

Example:

```csv
trial_id,session_id,question_text,instruction,label,answer,response_time,start_time,end_time
T001,S001,"Are you a student?","truth",0,"yes",1.42,0.00,5.00
T002,S001,"Are you looking at a screen?","lie",1,"no",1.89,6.00,11.00
```

A trial is one question shown to the participant. `instruction` can be "truth" or "lie". `label` is 0 for truth and 1 for lie. `response_time` is the time in seconds before the participant answered. `start_time` and `end_time` describe the trial time interval inside the session.

## gaze_samples.csv

Columns:
- sample_id
- trial_id
- timestamp
- gaze_x
- gaze_y
- pupil_left
- pupil_right
- blink
- fixation
- saccade
- validity

Example:

```csv
sample_id,trial_id,timestamp,gaze_x,gaze_y,pupil_left,pupil_right,blink,fixation,saccade,validity
1,T001,0.000,0.51,0.43,3.21,3.18,0,1,0,1
2,T001,0.016,0.52,0.44,3.22,3.17,0,1,0,1
```

Fields:
- sample_id: unique sample number
- trial_id: trial this sample belongs to
- timestamp: time inside the trial
- gaze_x: normalized horizontal gaze position
- gaze_y: normalized vertical gaze position
- pupil_left: left pupil diameter
- pupil_right: right pupil diameter
- blink: 0 or 1
- fixation: 0 or 1
- saccade: 0 or 1
- validity: 0 or 1, where 1 means valid sample

Each row in `gaze_samples.csv` belongs to one trial through `trial_id`. This makes it possible to connect raw gaze samples with the trial instruction, answer, response time, and label stored in `trials.csv`.

## Gaze coordinate format

Gaze coordinates are normalized from 0 to 1:
- gaze_x = 0.0 means left side of the screen
- gaze_x = 1.0 means right side of the screen
- gaze_y = 0.0 means top of the screen
- gaze_y = 1.0 means bottom of the screen

Example:

If the screen resolution is 1920x1080 and the gaze point is at pixel position x=960, y=540:

```text
gaze_x = 960 / 1920 = 0.5
gaze_y = 540 / 1080 = 0.5
```

## Processed data

Processed files will be stored in `data/processed` later.

Future processed files:
- windows.csv
- features.csv
- train.csv
- validation.csv
- test.csv

Do not create processed CSV files at this stage.
