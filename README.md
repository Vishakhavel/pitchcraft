# PitchCraft

Transforming Visions into Winning Pitches!

## About PitchCraft

PitchCraft is an AI-powered tool that evaluates the effectiveness of your pitches based on their purpose, goal, and target audience. It provides an objective assessment using well-established sales practices, empirical pitch history, and best body language practices.

## Team

- **Vishakhavel**: Full Stack Developer (Ex-Sony, Ex-Bank of America)
- **Shreyas**: Data and Machine Learning (Currently @ MooveAI, MS CS @ University of California)

## How It Works

1. **Initial Context**: Provide information about your audience, domain, and goal.
2. **Pitch Recording**: Capture a video of your pitch presentation.
3. **Audio Extraction**: Isolate and analyze the audio content.
4. **Body Language Assessment**: Evaluate gestures of elbows, wrists, and hands.
5. **Text Transcription**: Convert audio to text for further analysis.
6. **Content Evaluation**: Analyze the transcribed text using OpenAI API.
7. **Body Language Metrics**: Quantify non-verbal cues and their impact on the pitch.
8. **Fine-Tuning Results**:
   - External Dataset: Insights from a comprehensive sales pitches textbook.
   - Internal Dataset: Historical evaluations from our platform.

## What You Get

PitchCraft provides a comprehensive evaluation of your pitch, including:

- Objective Score
- General Feedback
- Projected Outcome
- Suggestions for Improvement
- Historical Comparison
- Sales Practice Advice
- Body Language Comments

## Output Format

The evaluation is provided in a structured format:

```json
{
  "score": float,
  "feedback": string,
  "projected_outcome": string,
  "suggestions": string,
  "comparison": string,
  "sales_strategies": string,
  "body_language_comments": string
}
```

---

PitchCraft - HockeyHacks AI 2024 Submission by Vishakhavel and Shreyas
