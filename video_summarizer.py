import yt_dlp
import whisper
import torch
from transformers import pipeline
from pytube import YouTube
import os

# Add FFmpeg to the PATH
ffmpeg_path = r"C:\VARUN\ffmpeg-2024-10-02-git-358fdf3083-essentials_build\ffmpeg-2024-10-02-git-358fdf3083-essentials_build\bin"  # Replace with your actual FFmpeg path
os.environ["PATH"] += os.pathsep + ffmpeg_path

class VideoSummarizer:
    def __init__(self):
        self.transcription_model = whisper.load_model("base")
        self.summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")
        self.qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        
    def download_audio(self, url):
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
        
    def transcribe_audio(self):
        result = self.transcription_model.transcribe("audio.mp3")
        return result["text"]
    
    def summarize_text(self, text):
        summary = self.summarization_model(text, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    
    def answer_question(self, context, question):
        answer = self.qa_model(question=question, context=context)
        return answer['answer']
    
    def process_video(self, url):
        print("Downloading audio...")
        self.download_audio(url)
        
        print("Transcribing audio...")
        transcription = self.transcribe_audio()
        
        print("Generating summary...")
        summary = self.summarize_text(transcription)
        
        return transcription, summary
    
def main():
    summarizer = VideoSummarizer()
    
    url = input("Enter the YouTube video URL: ")
    
    transcription, summary = summarizer.process_video(url)
    
    print("\nVideo Summary:")
    print(summary)
    
    while True:
        question = input("\nAsk a question about the video (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        answer = summarizer.answer_question(transcription, question)
        print("Answer:", answer)

if __name__ == "__main__":
    main()