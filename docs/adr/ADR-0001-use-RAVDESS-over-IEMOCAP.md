# ADR-0001: Use RAVDESS Instead of IEMOCAP

## Context
We have previously trained and experimented with an SER model trained on IEMOCAP and failed to reach anywhere above 70% validation accuracy on the dataset. IEMOCAP is old, noisy, and very small, making it very difficult for our model to learn the acoustic features needed for complex emotion recognition.

## Decision
We will train the tone 2.0 SER model solely on RAVDESS.

## Rationale
RAVDESS is more robust, larger, newer, and provided much more promising results in experimentation. 

## Alternatives Considered
- CMU-MOSEI: hard to get and weird licensing issues
- TESS: too few actors and single-gender

## Consequences
### Positive
- More accurate and robust model. 
  
## Negative
- Weird labeling scheme
- Much larger in raw size than IEMOCAP
- More preprocessing work neccessary when compared to IEMOCAP

## Follow-ups
- Decide what feature extractor to use
