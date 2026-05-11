# Project Definition

## Project title

Real-Time Eye-Tracking-Based Deception Risk Detection Using Neural Networks

Обнаружение риска обмана на основе real-time eye-tracking данных с использованием нейронных сетей
## Main goal

The goal of this project is to develop a machine learning model that analyzes real-time eye-tracking data and estimates the probability of deceptive behavior in a controlled experimental setting.

The system does not claim to universally detect lies. Instead, it provides a deception risk score based on gaze behavior, pupil changes, blink patterns, and fixation/saccade features.

## What the system detects

The system detects patterns that may be associated with deceptive or concealed responses during controlled tasks.

The target output is not a final judgment such as “truth” or “lie”, but a probability score:

- low deception risk
- medium deception risk
- high deception risk
- insufficient data

The system will not make final legal or psychological conclusions.
The system will not be used as an autonomous lie detector.
The system will not work in uncontrolled real-world environments at the first stage.

## In what scenario will model work

Statement-level deception task
## Input data

The system will use eye-tracking signals such as:

- gaze coordinates
- pupil diameter
- blink-related features
- fixation features
- saccade features
- missing-data indicators

## Output

The model will output a calibrated deception risk score.

Example:

Deception risk: 0.72  
Confidence: Medium  
Status: High risk

## Main limitation

The model is designed only for controlled experimental conditions. It should not be used as an autonomous lie detector in legal, employment, security, or medical decisions.

## Research basis

The project follows the findings of the article, which states that eye-tracking is promising for deception-related analysis, but real-time deployment requires causal preprocessing, subject-independent validation, missing-data handling, latency reporting, and careful interpretation of results.