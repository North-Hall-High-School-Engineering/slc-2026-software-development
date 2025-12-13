# Personal interpretations / context research

From huggingface, IEMOCAP only has a train split. This means that later down the road we'll have to manually split it (not a big deal).

Overall structure:
```
DatasetDict({
    train: Dataset({
        features: ['file', 'audio', 'frustrated', 'angry', 'sad', 'disgust', 'excited', 'fear', 'neutral', 'surprise', 'happy', 'EmoAct', 'EmoVal', 'EmoDom', 'gender', 'transcription', 'major_emotion', 'speaking_rate', 'pitch_mean', 'pitch_std', 'rms', 'relative_db'],
        num_rows: 10039
    })
})
```

## Overview of features
- file: useless 
- audio: raw np.array of audio data
- frustrated, angry, sad, disgust, excited, fear, neutral, surprise, happy [.00625, (1-.00625)]: percentage values dictating relative expression of a given emotion.
- EmoAct - emotion activation [1, 5]: how intense/activated the emotion is (1=calm, 5=excited).
- EmoVal - valence [1, 5]: how positive or negative the emotion feels (1=negative, 5=positive).
- EmoDom - dominance [1, 5]: how much control is expressed in the emotion (1=submissive, 5=dominant).
- gender: useless (for now)
- transcription: used in the future for multi-modal with RoBERTa
- major_emotion: string of the major emotion, used for discrete
- speaking_rate: syllables per second?
- pitch_mean: useless (for now)
- pitch_std: useless (for now)
- rms: root mean square?
- relative_db: relative volume, could also be useful

## The Big Question: Discrete or Continuous?
I think likely continuous, more nuance in predictions due to IEMOCAP's incredible relative emotion values.
