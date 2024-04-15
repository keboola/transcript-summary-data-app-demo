import threading
import queue
import time
from google.cloud import speech
import pydub
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.generative_models import GenerativeModel, Part, FinishReason
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["google_cloud"]
)

def generate_summary(content):
    vertexai.init(project="keboola-ai", location="us-central1", credentials=credentials)
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

    output_text = "".join(response.text for response in responses)
    return output_text

def start_audio_stream(webrtc_ctx, transcript_queue, stop_event):
    client = speech.SpeechClient(credentials=credentials)
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
            continue

        sound_chunk = pydub.AudioSegment.empty()
        for audio_frame in audio_frames:
            sound = pydub.AudioSegment(
                data=audio_frame.to_ndarray().tobytes(),
                sample_width=audio_frame.format.bytes,
                frame_rate=audio_frame.sample_rate,
                channels=len(audio_frame.layout.channels),
            )
            sound_chunk += sound

        if len(sound_chunk) > 0:
            sound_chunk = sound_chunk.set_channels(1).set_frame_rate(16000)
            audio_bytes = sound_chunk.raw_data
            requests = (speech.StreamingRecognizeRequest(audio_content=audio_bytes),)
            responses = client.streaming_recognize(config=streaming_config, requests=requests)
            for response in responses:
                for result in response.results:
                    if result.is_final:
                        transcript = result.alternatives[0].transcript
                        transcript_queue.put(transcript)

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
            transcript = transcript_queue.get(timeout=1)
            if transcript:
                print(f"Received transcript: {transcript}")  # Debugging print
            accumulated_text += " " + transcript

        except queue.Empty:
                print("No new transcripts received in the last 10 seconds.")
                continue  # Continue the while loop if no transcript is available

        if accumulated_text:
            print(f"Accumulated transcript for summary: {accumulated_text}")  # Debugging print
            summary = generate_summary(accumulated_text)
            if summary:
                print(f"Generated summary: {summary}")  # Debugging print
            summary_queue.put(summary)
            accumulated_text = ""
        time.sleep(15)  

def main():
    st.title("Real-time Speech Recognition and Summary Generation")
    transcript_queue = queue.Queue()
    summary_queue = queue.Queue()
    stop_event = threading.Event()

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=2048,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.relay.metered.ca:80"]}, {"urls":"turn:global.relay.metered.ca:80","username":"1b8aa9accf06c1594059c08b","credential":"dnDQUZM/2tgGULlF",}, {"urls":"turn:global.relay.metered.ca:80?transport=tcp","username":"1b8aa9accf06c1594059c08b","credential":"dnDQUZM/2tgGULlF",}, {"urls":"turn:global.relay.metered.ca:443","username":"1b8aa9accf06c1594059c08b","credential":"dnDQUZM/2tgGULlF",}, {"urls":"turns:global.relay.metered.ca:443?transport=tcp","username":"1b8aa9accf06c1594059c08b","credential":"dnDQUZM/2tgGULlF",}]},
        media_stream_constraints={"video": False, "audio": True},
    )

    audio_thread = threading.Thread(target=start_audio_stream, args=(webrtc_ctx, transcript_queue, stop_event))
    summary_thread = threading.Thread(target=summary_update, args=(transcript_queue, summary_queue, stop_event))

    if st.button("Start Recording"):
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