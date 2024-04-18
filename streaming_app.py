import queue
import threading
import sounddevice as sd
from google.cloud import speech
from google.oauth2 import service_account
import streamlit as st
import sys
import base64
import os
import time
import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.generative_models import GenerativeModel, Part, FinishReason

# Assuming your Google Cloud credentials are set up in Streamlit's secrets
#credentials_path = ""

#credentials = service_account.Credentials.from_service_account_file(credentials_path)
client = speech.SpeechClient()
    #credentials=credentials

image_path = os.path.dirname(os.path.abspath(__file__))

KBC_LOGO = image_path + "/static/keboola-gemini.png"
GC_LOGO = image_path + "/static/gemini.png"

def create_or_clear_file(file_path):
    """Create a new empty file or clear an existing file."""
    with open(file_path, 'w') as file:
        file.truncate()  

def append_to_file(file_path, data):
    """Append data to a file."""
    with open(file_path, 'a', encoding="utf-8") as file:
        file.write(data)

def read_file(file_path):
    """Read and return the contents of a file."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding="utf-8") as file:
            return file.readlines()
    return []

def audio_stream_generator(q):
    """Generator function that yields audio chunks from a queue."""
    while True:
        chunk = q.get()
        if chunk is None:  # Use None as a signal to end the stream.
            break
        yield speech.StreamingRecognizeRequest(audio_content=chunk)

def stream_audio(transcript_queue, stop_event, device_index=None):
    while not stop_event.is_set():
        audio_q = queue.Queue(maxsize=10)
        start_time = time.time()
    
        def audio_callback(indata, frames, time, status):
            if status:
                print("Audio Input Error:", status, file=sys.stderr)
            audio_q.put(bytes(indata))

        with sd.RawInputStream(callback=audio_callback, dtype='int16', channels=1, samplerate=16000, device=device_index):
            requests = audio_stream_generator(audio_q)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code='en-US',
            )
            streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

            try:
                responses = client.streaming_recognize(config=streaming_config, requests=requests)
                for response in responses:
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    if elapsed_time > 240:
                        break
                    
                    if response.error.code:
                        print("Google Speech API Error:", response.error.message)
                        break

                    for result in response.results:
                        if result.is_final:
                            transcript = result.alternatives[0].transcript
                            transcript_queue.put(transcript)
                            print("Transcript updated:", transcript)  # Debugging print

                    if stop_event.is_set():
                        print("Stop event triggered.")
                        break

            except Exception as e:
                print("Exception during streaming:", e)
            finally:
                audio_q.put(None)  # Signal the generator to terminate

        time.sleep(1)

def generate_summary(content):
    vertexai.init(project="keboola-ai", location="us-central1", 
                  #credentials=credentials
                  )
    model = GenerativeModel("gemini-1.5-pro-preview-0409")
   
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }
    
    responses = model.generate_content(
        contents=f"""
        Create a concise (1-2 sentences) summary from given transcript, and translate it to Romanian. 
        Write the summary in the third person, do NOT mention the transcript in the summary.  
        Include a dash at the beginning so that it can be used as a bullet point.
        When you don't get any script, don't hallucinate. 
        
        Return only the translated summary.

        Transcript: 
        {content}
        """,
        generation_config=generation_config,
        stream=True,
    )

    output_text = "".join(response.text for response in responses)
    return output_text


def stream_summaries(summary_queue):
    while True:
        summary = summary_queue.get()
        if summary is None:
            break
        yield summary

def summary_update(transcript_queue, summary_queue, stop_event, file_path):
    accumulated_text = ""
    
    while not stop_event.is_set():
        try:
            transcript = transcript_queue.get_nowait()
            if transcript:
                print(f"Received transcript: {transcript}")  # Debugging print
            accumulated_text += " " + transcript
        except queue.Empty:
            if accumulated_text:
                print(f"Accumulated transcript for summary: {accumulated_text}")  # Debugging print
                summary = generate_summary(accumulated_text)
                if summary:
                    print(f"Generated summary: {summary}")  # Debugging print
                    
                append_to_file(file_path, summary)
                summary_queue.put(summary)
                accumulated_text = ""
            time.sleep(30)

def main():
    
    logo_html = f'''
    <div style="display: flex; justify-content: flex-end;">
        <img src="data:image/png;base64,{base64.b64encode(open(KBC_LOGO, "rb").read()).decode()}" style="height: 100px;">
        <br><br>
    </div>
    '''
    st.markdown(f"{logo_html}", unsafe_allow_html=True)
    
    st.title("Real-time Speech Recognition and Summary Generation")
    summaries_file_path = 'summaries.txt'

    transcript_queue = queue.Queue()
    summary_queue = queue.Queue()
    stop_event = threading.Event()

    transcription_thread = None
    summary_thread = None
    
    st.markdown("####")
    
    col1, col2 = st.columns(2)    
    start_button = col1.button("Start", use_container_width=True)
    stop_button = col2.button("Stop", use_container_width=True)

    if start_button:
        create_or_clear_file(summaries_file_path)
        stop_event.clear()
        transcription_thread = threading.Thread(
            target=stream_audio, args=(transcript_queue, stop_event))
        summary_thread = threading.Thread(
            target=summary_update, args=(transcript_queue, summary_queue, stop_event, summaries_file_path))

        transcription_thread.start()
        summary_thread.start()

        st.success("Recording and processing started.")
        st.subheader("Summaries")
        st.write_stream(stream_summaries(summary_queue))

    if stop_button:
        if transcription_thread and transcription_thread.is_alive():
            stop_event.set()
            transcription_thread.join()
        if summary_thread and summary_thread.is_alive():
            summary_thread.join()
        
        st.success("Recording and processing stopped.")
        st.subheader("Summaries")
        summaries = read_file(summaries_file_path)
        for summary in summaries:
            st.write(summary)
        
if __name__ == "__main__":
    main()