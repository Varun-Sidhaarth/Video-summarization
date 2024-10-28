from flask import Flask, request, jsonify, render_template_string
import yt_dlp
import whisper
from transformers import pipeline
import os
import logging
import google.generativeai as genai
from config import GOOGLE_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add FFmpeg to the PATH
ffmpeg_path = r"C:\ffmpeg\bin"  # Replace with your actual FFmpeg path
os.environ["PATH"] += os.pathsep + ffmpeg_path

# Configure Google AI
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)

class VideoSummarizer:
    def __init__(self):
        self.transcription_model = whisper.load_model("base")
        self.summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
    def download_audio(self, url):
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': 'audio.%(ext)s',
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            logger.info("Audio downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading audio: {str(e)}")
            raise
        
    def transcribe_audio(self):
        try:
            result = self.transcription_model.transcribe("audio.mp3")
            logger.info("Audio transcribed successfully")
            return result["text"]
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise
    
    def summarize_text(self, text):
        try:
            max_chunk_length = 1000
            chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            
            summaries = []
            for chunk in chunks:
                summary = self.summarization_model(chunk, max_length=150, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            
            final_summary = " ".join(summaries)
            logger.info("Text summarized successfully")
            return final_summary
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            raise
    
    def answer_question(self, context, question):
        try:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise
    
    def process_video(self, url):
        try:
            self.download_audio(url)
            transcription = self.transcribe_audio()
            summary = self.summarize_text(transcription)
            return transcription, summary
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

summarizer = VideoSummarizer()

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    answer = None
    if request.method == 'POST':
        if 'url' in request.form:
            url = request.form.get('url')
            if url:
                try:
                    transcription, summary = summarizer.process_video(url)
                    result = {
                        "transcription": transcription,
                        "summary": summary
                    }
                except Exception as e:
                    result = {"error": str(e)}
        elif 'question' in request.form:
            question = request.form.get('question')
            transcription = request.form.get('transcription')
            if question.lower() == 'quit':
                return render_template_string(html, result=None, answer="Thank you for using the Q&A system!")
            if question and transcription:
                try:
                    answer = summarizer.answer_question(transcription, question)
                except Exception as e:
                    answer = f"Error: {str(e)}"

    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Video Summarizer and Q&A</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
            form { margin-bottom: 20px; }
            input[type="text"], input[type="submit"] { padding: 5px; margin-right: 10px; }
            input[type="text"] { width: 300px; }
            pre { background-color: #f4f4f4; padding: 10px; white-space: pre-wrap; word-wrap: break-word; }
        </style>
    </head>
    <body>
        <h1>Video Summarizer and Q&A (Powered by Gemini AI)</h1>
        <form method="post">
            <input type="text" name="url" placeholder="Enter YouTube URL" required>
            <input type="submit" value="Process Video">
        </form>
        {% if result %}
            {% if result.error %}
                <h2>Error:</h2>
                <p>{{ result.error }}</p>
            {% else %}
                <h2>Summary:</h2>
                <p>{{ result.summary }}</p>
                <h2>Ask a Question (Gemini AI)</h2>
                <p>Type 'quit' to exit the Q&A session.</p>
                <form method="post">
                    <input type="text" name="question" placeholder="Enter your question" required>
                    <input type="hidden" name="transcription" value="{{ result.transcription }}">
                    <input type="submit" value="Ask">
                </form>
            {% endif %}
        {% endif %}
        {% if answer %}
            <h2>Answer (Gemini AI):</h2>
            <p>{{ answer }}</p>
            <h2>Ask a Question (Gemini AI)</h2>
            <p>Type 'quit' to exit the Q&A session.</p>
            <form method="post">
                <input type="text" name="question" placeholder="Enter your question" required>
                <input type="hidden" name="transcription" value="{{ result.transcription }}">
                <input type="submit" value="Ask">
            </form>
        {% endif %}
    </body>
    </html>
    """
    return render_template_string(html, result=result, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)