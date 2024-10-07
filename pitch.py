import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import os
import cv2
import mediapipe as mp
import numpy as np
import json
import time
import logging
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import os
import cv2
import mediapipe as mp
import numpy as np
import json
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedPitchGrader:
    def __init__(self, api_key, mock_dataset_path):
        openai.api_key = api_key
        self.mock_dataset = pd.read_csv(mock_dataset_path)
        # add context stuff
        # self.goal = goal
        # self.target_audience = target_audience
        # self.industry = industry

        self.sales_dataset = self.load_sales_dataset()
        self.vectorizer = TfidfVectorizer()
        self.combined_data = self.combine_datasets()
        self.vectors = self.vectorizer.fit_transform(self.combined_data['text'])
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_face_detection = mp.solutions.face_detection

    def load_sales_dataset(self):
        dataset = load_dataset("goendalf666/sales-textbook_for_convincing_and_selling")
        return pd.DataFrame(dataset['train'])

    def combine_datasets(self):
        mock_data = self.mock_dataset.copy()
        mock_data['text'] = mock_data['feedback'] + " " + mock_data['projected_outcome'] + " " + mock_data['actual_outcome']
        mock_data['source'] = 'mock'

        sales_data = self.sales_dataset.copy()
        sales_data['source'] = 'sales_textbook'
        sales_data = sales_data.rename(columns={'text': 'text'})

        combined = pd.concat([mock_data, sales_data], ignore_index=True)
        return combined

    def get_relevant_content(self, pitch, n=5):
        pitch_vector = self.vectorizer.transform([pitch])
        similarities = cosine_similarity(pitch_vector, self.vectors).flatten()
        top_indices = similarities.argsort()[-n:][::-1]
        return self.combined_data.iloc[top_indices]

    def generate_enhanced_prompt(self, pitch, context, relevant_content, video_analysis):
        prompt = f"""As an AI pitch expert trained on extensive sales techniques and real pitch outcomes, analyze the following pitch. If the pitch is lesser than 10 seconds, and if it is bad, be very strict and reduce the score:

Pitch Context:
- Target Audience: {context['target_audience']}
- Industry: {context['industry']}
- Goal: {context['goal']}

Pitch Transcript:
{pitch}

Relevant Information:
"""
        for _, row in relevant_content.iterrows():
            if row['source'] == 'mock':
                prompt += f"Similar Pitch (ID: {row['pitch_id']}, Score: {row['score']}/10):\n"
                prompt += f"Feedback: {row['feedback']}\n"
                prompt += f"Projected Outcome: {row['projected_outcome']}\n"
                prompt += f"Actual Outcome: {row['actual_outcome']}\n\n"
            else:
                prompt += f"Sales Technique:\n{row['text'][:500]}...\n\n"

        prompt += f"""
Video Analysis:
{video_analysis}

Based on the given pitch, the context, the similar pitches with their outcomes, the sales techniques, and the video analysis, provide:
1. An overall score out of 10
2. Detailed feedback on strengths and weaknesses
3. A projected outcome that specifically incorporates the pitch context (target audience, industry, and goal)
4. Suggestions for improvement, incorporating relevant sales techniques
5. Explanation of how this pitch compares to similar pitches, considering their actual outcomes
6. Recommendations for applying specific sales strategies from the textbook to enhance the pitch
7. Comments on the presenter's body language and presence, based on the video analysis

Format your response as a JSON object with the following keys:
{{
    "score": float,
    "feedback": string,
    "projected_outcome": string,
    "suggestions": string,
    "comparison": string,
    "sales_strategies": string,
    "body_language_comments": string
}}

Ensure that the "projected_outcome" explicitly mentions the target audience ({context['target_audience']}), industry ({context['industry']}), and goal ({context['goal']}) from the pitch context. If there is a lack of audio transcription or if the pitch is insulting, make the "score" lesser than 5. Be very strict in the evaluation.
"""
        return prompt

    def transcribe_audio(self, audio_file_path):
        audio = AudioSegment.from_file(audio_file_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
            audio.export(temp_wav.name, format="wav")
            wav_path = temp_wav.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        
        os.unlink(wav_path)

        try:
            transcript = recognizer.recognize_google(audio_data)
            return transcript
        except sr.UnknownValueError:
            return "Speech recognition could not understand the audio"
        except sr.RequestError as e:
            return f"Could not request results from the speech recognition service; {e}"

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        person_detected_frames = 0
        pose_data = []
        warnings = []

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            with self.mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    pose_results = pose.process(rgb_frame)
                    
                    face_results = face_detection.process(rgb_frame)

                    if pose_results.pose_landmarks or (face_results.detections and len(face_results.detections) > 0):
                        person_detected_frames += 1
                        if pose_results.pose_landmarks:
                            landmarks = pose_results.pose_landmarks.landmark
                            pose_data.append([
                                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y
                            ])
                    else:
                        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        warnings.append(f"Warning: No person detected at {current_time:.2f} seconds")

        cap.release()

        pose_analysis = self.analyze_pose_data(pose_data)

        person_detection_rate = person_detected_frames / total_frames
        overall_analysis = self.generate_body_language_analysis(person_detection_rate, pose_analysis, warnings)

        return overall_analysis

    def analyze_pose_data(self, pose_data):
        if not pose_data:
            return "No pose data available for analysis."

        pose_array = np.array(pose_data)
        
        shoulder_movement = np.std(pose_array[:, 0:2])
        arm_movement = np.mean(np.std(pose_array[:, 2:], axis=0))

        analysis = ""
        if shoulder_movement < 0.02:
            analysis += "The presenter appears very still, which might indicate nervousness or lack of confidence. "
        elif shoulder_movement > 0.05:
            analysis += "The presenter shows significant upper body movement, suggesting enthusiasm or nervousness. "
        else:
            analysis += "The presenter's posture seems balanced and confident. "

        if arm_movement < 0.05:
            analysis += "There's minimal arm movement, which might make the presentation less engaging. "
        elif arm_movement > 0.1:
            analysis += "The presenter uses a lot of hand gestures, which can be engaging but might also be distracting. "
        else:
            analysis += "The amount of gesticulation seems appropriate and engaging. "

        return analysis

    def generate_body_language_analysis(self, person_detection_rate, pose_analysis, warnings):
        analysis = f"Person Detection Rate: {person_detection_rate*100:.2f}%\n\n"
        analysis += "Body Language Analysis:\n"
        analysis += pose_analysis + "\n\n"
        
        if warnings:
            analysis += "Warnings:\n"
            analysis += "\n".join(warnings) + "\n\n"
        
        if person_detection_rate < 0.9:
            analysis += "Overall: The presenter was not consistently visible in the frame. This could negatively impact the effectiveness of the pitch.\n"
        else:
            analysis += "Overall: The presenter maintained a good presence in the frame throughout the pitch.\n"

        return analysis

    def retry_request(self, prompt, retries=5):
        """ Retry mechanism for handling rate limit and transient errors. """
        for attempt in range(retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
        {"role": "system", "content": "You are an expert in evaluating sales pitches."},
        {"role": "user", "content": prompt}
    ],
                    temperature=0.7,
                    max_tokens=2000,
                )

                return response
            # except openai.error.RateLimitError as e:
            #     logging.warning(f"Rate limit exceeded. Attempt {attempt+1}/{retries}. Retrying in {2 ** attempt} seconds.")
            #     time.sleep(2 ** attempt)
            # except openai.error.OpenAIError as e:
            #     logging.error(f"OpenAI API Error: {str(e)}")
            #     break  # Exit the loop on non-retryable errors

            except Exception as e:
                logging.error(f"Error: {str(e)}")
                break  # Exit the loop on non-retryable errors
        return None

    def extract_audio(self, video_path):
        temp_audio_path = tempfile.mktemp(suffix=".wav")
        os.system(f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {temp_audio_path}")
        return temp_audio_path

    def grade_pitch(self, context, video_path):
        logging.info("Extracting audio from video...")
        audio_path = self.extract_audio(video_path)

        logging.info("Transcribing audio...")
        pitch = self.transcribe_audio(audio_path)
        
        logging.info("Analyzing video...")
        video_analysis = self.analyze_video(video_path)
        
        logging.info("Finding relevant content...")
        relevant_content = self.get_relevant_content(pitch)
        
        logging.info("Generating enhanced prompt...")
        prompt = self.generate_enhanced_prompt(pitch, context, relevant_content, video_analysis)

        logging.info("Requesting response from OpenAI API...")
        response = self.retry_request(prompt)
        
        # Clean up temporary audio file
        os.remove(audio_path)

        if response:
            return json.loads(response['choices'][0]['message']['content'])
        else:
            logging.error("Failed to retrieve a response from OpenAI API after multiple attempts.")
            return {"error": "Failed to retrieve a response from the API"}

# Example usage

# main function
# def main(filename, goal, domain, target_audience):

# Example usage
# if __name__ == "__main__":
    # main()


def check(file, goal, domain, target_audience, file_path):
    print('inside the check function!!!', file, goal, domain, target_audience, file_path)
    api_key = "your_api_key"
    mock_dataset_path = "large_mock_pitch_dataset.csv"
    pitch_grader = EnhancedPitchGrader(api_key, mock_dataset_path)

    
    # context = {
    #     'target_audience': 'investors',
    #     'industry': 'tech',
    #     'goal': 'raise $1M in funding'
    # }
    context = {'goal' :goal, 'industry': domain, 'target_audience' : target_audience}

    # video_path = "uploads/video_file.mp4"
    # video_path = "uploads/video_file.mp4"
    # video_path = os.path.join('uploads', file_path)

    # result = pitch_grader.grade_pitch(context, video_path)
    result = pitch_grader.grade_pitch(context, file_path)
    
    print(result)
    # return the result
    return result
