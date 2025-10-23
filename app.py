# booking_frontend_fast.py

import streamlit as st
import requests
import json
from datetime import time, date, timedelta
import base64
from audio_recorder_streamlit import audio_recorder

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000" 
STT_TTS_ENDPOINT = "http://127.0.0.1:8000"

# --- Page Setup ---
st.set_page_config(page_title="Book Your Session", layout="centered")
st.title("Find and Book Your Expert Session")

# --- Session State Initialization ---
if "booking_mode" not in st.session_state: st.session_state.booking_mode = None
if 'manual_booking_details' not in st.session_state: st.session_state.manual_booking_details = None
if 'audio_to_process' not in st.session_state: st.session_state.audio_to_process = None
if 'audio_player_key' not in st.session_state: st.session_state.audio_player_key = 0

# --- UI Functions ---
def show_initial_selection():
    st.subheader("How would you like to book?")
    c1, c2 = st.columns(2)
    if c1.button("📝 Book Manually", use_container_width=True):
        st.session_state.booking_mode = "manual"
        st.rerun()
    if c2.button("🎙️ Book with Voice", use_container_width=True):
        st.session_state.booking_mode = "audio"
        st.session_state.messages = []
        st.session_state.conversation_state = {}
        st.rerun()

def show_manual_booking_form():
    st.header("Manual Booking Form")
    st.markdown("Fill in the details below to schedule your appointment.")
    EXPERT_NAMES = ["Dr. Ranjit Sharma", "Dr. ravina Kumar", "Dr. Shyam Singh", "Dr. Shadab Patel", "Dr. Naik Das", "Dr. Vinay Gupta", "Dr. Jagadesh Devi", "Dr. Dipanshi Mehta", "Dr. Sumitha Reddy", "Dr. Arjun Joshi"]
    SESSION_TYPES = ["Yoga", "Stress Relief", "Mindfulness", "Life Coaching", "Nutritional Advice"]
    APPOINTMENT_TYPES = ["Video Call", "Audio Call"]
    REPEAT_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    REMINDER_OPTIONS = ["5 minutes before", "10 minutes before", "15 minutes before"]
    TIME_SLOTS = [time(h, m) for h in range(9, 18) for m in (0, 30)]

    with st.form("booking_form"):
        st.subheader("Booking Details")
        col1, col2 = st.columns(2)
        with col1:
            user_name = st.text_input("Your Name*", help="Please enter your full name.")
            expert_name = st.selectbox("Find an Expert*", EXPERT_NAMES, help="Choose an expert.")
            session_date = st.date_input("Select a Date*", min_value=date.today())
            time_slot = st.selectbox("Select a Time Slot*", options=TIME_SLOTS, format_func=lambda t: t.strftime('%I:%M %p'))
        with col2:
            session_type = st.selectbox("Session Type*", SESSION_TYPES)
            appointment_type = st.selectbox("Appointment Type*", APPOINTMENT_TYPES)
            reminder = st.selectbox("Set a Reminder*", REMINDER_OPTIONS)
        
        st.subheader("Repeating Session (Optional)")
        repeat_on = st.multiselect("Repeat On", REPEAT_DAYS)
        end_date = st.date_input("End Date", min_value=session_date + timedelta(days=1) if session_date else date.today())
        
        st.markdown("---")
        submitted = st.form_submit_button("Book My Session")

    if submitted:
        if not user_name: st.error("Please fill in your name.")
        else:
            with st.spinner("Booking your session..."):
                booking_payload = {
                    "user_id": "user_manual_123", "user_name": user_name, "expert_name": expert_name,
                    "session_date": session_date.isoformat(), "time_slot": time_slot.isoformat(),
                    "session_type": session_type, "appointment_type": appointment_type,
                    "repeat_on": repeat_on or None, "end_date": end_date.isoformat() if repeat_on else None,
                    "reminder": reminder
                }
                try:
                    response = requests.post(f"{BACKEND_URL}/book-session", json=booking_payload)
                    if response.status_code == 200: st.session_state.manual_booking_details = response.json()
                    else: st.error(f"Booking Failed: {response.text}")
                except Exception as e: st.error(f"Connection Error: {e}")

    if st.session_state.manual_booking_details:
        st.success("🎉 Your session has been booked successfully!")
        st.json(st.session_state.manual_booking_details)

def process_conversation_stream(user_input=""):
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": "", "audio_data": None, "booking_details": None})
    st.session_state.audio_player_key += 1
    st.rerun()

def play_audio(audio_data_hex, autoplay=False):
    if not audio_data_hex: return None
    audio_bytes = bytes.fromhex(audio_data_hex)
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    autoplay_str = "autoplay" if autoplay else ""
    audio_html = f"""
        <audio controls {autoplay_str}>
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        </audio>
    """
    return audio_html

def show_audio_booking_interface():
    st.header("Conversational Audio Booking")

    if st.session_state.audio_to_process:
        audio_bytes = st.session_state.audio_to_process
        st.session_state.audio_to_process = None
        with st.spinner("Transcribing your voice..."):
            try:
                files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
                stt_res = requests.post(f"{STT_TTS_ENDPOINT}/transcribe/", files=files, timeout=30)
                stt_res.raise_for_status()
                
                # --- CHANGE 1: Capture both text and duration ---
                response_data = stt_res.json()
                transcribed_text = response_data.get('transcribed_text')
                stt_duration = response_data.get('stt_duration') # Get the duration
                
                if transcribed_text and "Could not understand" not in transcribed_text:
                    # Store the duration in session_state to use it in the next step
                    if stt_duration is not None:
                         st.session_state.stt_duration_for_next_request = stt_duration
                    process_conversation_stream(user_input=transcribed_text)
                else:
                    st.warning("Sorry, I didn't catch that. Could you please say it again?")
            except Exception as e: st.error(f"Error during transcription: {e}")
            # try:
            #     files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
            #     stt_res = requests.post(f"{STT_TTS_ENDPOINT}/transcribe/", files=files, timeout=30)
            #     stt_res.raise_for_status()
            #     transcribed_text = stt_res.json().get('transcribed_text')
            #     if transcribed_text and "Could not understand" not in transcribed_text:
            #         process_conversation_stream(user_input=transcribed_text)
            #     else:
            #         st.warning("Sorry, I didn't catch that. Could you please say it again?")
            # except Exception as e: st.error(f"Error during transcription: {e}")
    
    messages = st.session_state.get("messages", [])
    for i, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            text_placeholder = st.empty()
            audio_placeholder = st.empty()
            
            text_placeholder.markdown(msg["content"])
            if msg.get("booking_details"):
                st.success("Booking Confirmed!")
                st.json(msg["booking_details"])

            # Logic for assistant messages
            if msg["role"] == "assistant":
                is_last_message = (i == len(messages) - 1)

                # If it's the last message and has no content yet, stream into it
                if is_last_message and not msg["content"]:
                    try:
                        user_input = st.session_state.messages[-2].get("content", "") if len(st.session_state.messages) > 1 else ""
                        
                        # --- CHANGE 2: Add stt_duration to the payload ---
                        payload = {
                            "user_input": user_input, 
                            "state": st.session_state.conversation_state,
                            # Get the duration from session_state and remove it so it's not reused
                            "stt_duration": st.session_state.pop('stt_duration_for_next_request', None) 
                        }
                        
                        response = requests.post(f"{BACKEND_URL}/converse-booking-stream", json=payload, stream=True, timeout=60)
                    
                    # try:
                    #     user_input = st.session_state.messages[-2].get("content", "") if len(st.session_state.messages) > 1 else ""
                    #     payload = {"user_input": user_input, "state": st.session_state.conversation_state}
                        
                    #     response = requests.post(f"{BACKEND_URL}/converse-booking-stream", json=payload, stream=True, timeout=60)
                        response.raise_for_status()
                        
                        for line in response.iter_lines():
                            if line:
                                data = json.loads(line.decode('utf-8'))
                                
                                if data.get("type") == "text_delta":
                                    msg["content"] += data["content"]
                                    text_placeholder.markdown(msg["content"] + " ▌")
                                
                                elif data.get("type") == "full_audio":
                                    msg["audio_data"] = data["content"]
                                    audio_html = play_audio(msg["audio_data"], autoplay=True)
                                    if audio_html:
                                        audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
                                
                                elif data.get("type") == "state_update":
                                    st.session_state.conversation_state = json.loads(data["content"])

                                elif data.get("type") == "final_confirmation":
                                    msg["content"] = data["text"]
                                    msg["audio_data"] = data["audio"]
                                    msg["booking_details"] = json.loads(data["booking_details"])
                                    st.session_state.conversation_state = json.loads(data["updated_state"])
                                    text_placeholder.markdown(msg["content"])
                                    audio_html = play_audio(msg["audio_data"], autoplay=True)
                                    if audio_html:
                                        audio_placeholder.markdown(audio_html, unsafe_allow_html=True)

                                elif data.get("type") == "error":
                                    st.error(f"An error occurred in the backend: {data['content']}")

                        text_placeholder.markdown(msg["content"])

                    except Exception as e:
                        error_message = f"A streaming error occurred: {e}"
                        st.error(error_message)
                        msg["content"] = "Sorry, I encountered a problem. Please try again."
                        text_placeholder.markdown(msg["content"])
                
                # If it's a past message that already has audio data, just show the player without autoplay
                elif msg.get("audio_data"):
                    audio_html = play_audio(msg["audio_data"], autoplay=False)
                    if audio_html:
                        audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
    
    if not st.session_state.get("messages"):
        with st.spinner("Assistant is starting..."):
            try:
                # This is the new static welcome message
                welcome_text = (
                    "Hello! I'm your friendly assistant here to help you book an appointment. Let's get started!\n\n"
                    "Could you please tell me which expert you'd like to book your session with? Here are our available experts: "
                    "Dr. Anjali Singh, and Dr. Amit Patel."
                )

                # Call the TTS service to get the audio for the welcome message
                tts_payload = {"text": welcome_text, "voice": "af_heart"}
                tts_res = requests.post(f"{STT_TTS_ENDPOINT}/tts", json=tts_payload, timeout=30)
                tts_res.raise_for_status()
                audio_content_hex = tts_res.content.hex()

                # Manually add the first, static message to the chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": welcome_text,
                    "audio_data": audio_content_hex,
                    "booking_details": None
                })

                # Pre-populate the backend conversation state with the static message
                # This gives the AI context for the user's first actual reply
                if 'conversation_state' not in st.session_state or not st.session_state.conversation_state:
                     st.session_state.conversation_state = {}
                st.session_state.conversation_state['conversation_history'] = [
                    {"role": "assistant", "content": welcome_text}
                ]
                
                st.rerun()

            except Exception as e:
                st.error(f"Failed to start the assistant. Please check the connection and try again. Error: {e}")

    is_booking_complete = st.session_state.get('conversation_state', {}).get('booking_complete', False)
    if not is_booking_complete:
        st.markdown("---")
        recorded_audio = audio_recorder(text="Click to speak", pause_threshold=2.0)
        if recorded_audio:
            st.session_state.audio_to_process = recorded_audio
            st.rerun()

# --- Main App Router ---
if st.session_state.booking_mode is None:
    show_initial_selection()
else:
    if st.button("⬅️ Change Booking Method"):
        st.session_state.booking_mode = None
        st.session_state.manual_booking_details = None
        st.rerun()
    
    if st.session_state.booking_mode == "manual":
        show_manual_booking_form()
    elif st.session_state.booking_mode == "audio":
        show_audio_booking_interface()