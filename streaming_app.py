import numpy as np
import threading
import queue
import time
from google.cloud import speech
import streamlit as st
import av
import logging
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.generative_models import GenerativeModel, Part, FinishReason

def generate_summary(content):
    if not content.strip():
        return "- No content provided for summarization."
    
    vertexai.init(project="keboola-ai", location="us-central1")
    model = generative_models.GenerativeModel("gemini-1.5-pro-preview-0409")

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }

    responses = model.generate_content(
        contents=f"""
        Create a concise (1-2 sentences) summary from given transcript. 
        Write the summary in the third person, do NOT mention the transcript in the summary.  
        Include a dash at the beginning so that it can be used as a bullet point.

        Transcript: 
        {content}
        """,
        generation_config=generation_config,
        stream=True,
    )

    output_text = "".join(response.text for response in responses if response.text)
    if not output_text:
        return "- Unable to generate summary due to insufficient data."
    return output_text

def start_audio_stream(webrtc_ctx, transcript_queue, stop_event):
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US',
        enable_automatic_punctuation=True
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)
    
    requests = []  # This will store ongoing audio requests
    
    while not stop_event.is_set() and webrtc_ctx.state.playing:
        try:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=5)
        except queue.Empty:
            print("No audio frames received.")
            continue
        
        for frame in audio_frames:
            audio_samples = np.frombuffer(frame.to_ndarray().tobytes(), dtype=np.int16)
            
            if frame.layout.name == 'stereo':
                audio_samples = audio_samples.reshape(-1, 2).mean(axis=1).astype(np.int16)  # Stereo to mono
            
            audio_bytes = audio_samples.tobytes()
            requests.append(speech.StreamingRecognizeRequest(audio_content=audio_bytes))
        
        if requests:
            responses = client.streaming_recognize(config=streaming_config, requests=requests)
            process_responses(responses, transcript_queue)
            requests = []  # Reset requests after sending

def process_responses(responses, transcript_queue):
    for response in responses:
        for result in response.results:
            if result.is_final:
                transcript = result.alternatives[0].transcript
                if transcript:
                    transcript_queue.put(transcript)
                    print("Transcript added to queue:", transcript)
                else:
                    print("Received final result with empty transcript.")


def stream_transcripts(transcript_queue):
    while True:
        transcript = transcript_queue.get()
        if transcript is None:
            break
        yield transcript

def stream_summaries(summary_queue):
    while True:
        summary = summary_queue.get()
        if summary is None:
            break
        yield summary

def summary_update(transcript_queue, summary_queue, stop_event):
    accumulated_text = ""
    while not stop_event.is_set():
        try:
            transcript = transcript_queue.get(timeout=0)
            if transcript:
                accumulated_text += " " + transcript
                print("Accumulated transcript:", accumulated_text)
        except queue.Empty:
            print("Transcript queue is empty, checking again.")
            continue
        
        if accumulated_text.strip():
            summary = generate_summary(accumulated_text.strip())
            if summary:
                summary_queue.put(summary)
                print("Summary added to queue:", summary)
                accumulated_text = ""
            time.sleep(10)
def main():
    st.title("Real-time Speech Recognition and Summary Generation")
    transcript_queue = queue.Queue()
    summary_queue = queue.Queue()
    stop_event = threading.Event()

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=4096,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True}
    )

    audio_thread = threading.Thread(target=start_audio_stream, args=(webrtc_ctx, transcript_queue, stop_event))
    summary_thread = threading.Thread(target=summary_update, args=(transcript_queue, summary_queue, stop_event))

    if st.button("Start Recording"):
        print("Starting threads...")
        stop_event.clear()
        audio_thread.start()
        summary_thread.start()
        st.success("Recording started")
        st.subheader("Summaries")
        st.write_stream(stream_summaries(summary_queue))

    if st.button("Stop Recording"):
        stop_event.set()
        transcript_queue.put(None)  # Signal the generator to terminate
        summary_queue.put(None)
        audio_thread.join()
        summary_thread.join()
        st.success("Recording stopped")

if __name__ == "__main__":

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    main()
