import numpy as np
import threading
import queue
import time
from google.cloud import speech
import streamlit as st
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


def start_audio_stream(webrtc_ctx, transcript_queue, stop_event):
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US',
        enable_automatic_punctuation=True
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    while not stop_event.is_set() and webrtc_ctx.state.playing:
        try:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            print("No audio frames received.")
            continue

        if not audio_frames:
            print("No audio frames to process.")
            continue

        audio_samples = np.concatenate([
            np.frombuffer(frame.to_ndarray().tobytes(), dtype=np.int16) for frame in audio_frames
        ])

        # Ensure it is mono
        if len(audio_frames[0].layout.channels) > 1:
        # Assuming the input is stereo and channels are interleaved
            audio_samples = audio_samples.reshape(-1, 2).mean(axis=1).astype(np.int16)

        # Debug: Check the length and content of the audio_samples array
        print(f"Audio samples array length: {len(audio_samples)}")
        if len(audio_samples) == 0:
            print("Empty audio samples array.")
            continue

        # Convert numpy array to bytes and send to Google Speech API
        audio_bytes = audio_samples.tobytes()
        print(f"Audio bytes length: {len(audio_bytes)}")

        # Modify this part to change the buffer size
        if len(audio_samples) > 0:
            # Example: Send audio data in chunks of approximately 1 second (16000 samples for 16000 Hz audio)
            chunk_size = 16000  # Adjust size based on experimentation
            for start in range(0, len(audio_samples), chunk_size):
                end = start + chunk_size
                audio_chunk = audio_samples[start:end]
                audio_bytes = audio_chunk.tobytes()
                if len(audio_bytes) > 0:
                    requests = [speech.StreamingRecognizeRequest(audio_content=audio_bytes)]
                    responses = client.streaming_recognize(config=streaming_config, requests=requests)
                    process_responses(responses, transcript_queue)
                else:
                    print("Generated empty audio bytes.")



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
            transcript = transcript_queue.get(timeout=5)
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
        audio_receiver_size=1024,
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
    main()
