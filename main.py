# booking_backend.py

import os
import uuid
import json
from typing import List, Optional, AsyncGenerator, Sequence, Annotated
from operator import add as add_messages
import asyncio
import wave
import io
import torch
import time

# --- FastAPI and Pydantic Imports ---
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError
from datetime import date
from dateutil.parser import parse as parse_date
from datetime import time as time_obj
from fastapi import Response # Make sure to import Response from fastapi

# --- Voice and Audio Imports ---
import numpy as np
import warnings
from faster_whisper import WhisperModel
from kokoro import KPipeline

# --- LangChain and LangGraph Imports ---
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==================== Configuration and Setup ====================
MISTRAL_API_KEY = "Enter Your API Key Here"
MISTRAL_LLM_ENDPOINT = "Enter your endpoint here"

# --- Voice Model Initialization ---
try:
    print("Initializing voice models...")
    stt_model = WhisperModel("small", device="cpu", compute_type="int8")
    tts_pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    print("Voice models initialized successfully.")
except Exception as e:
    print(f"Error initializing voice models: {e}. Voice functionality will be disabled.")
    stt_model = None
    tts_pipeline = None

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "af_heart"

# --- Data Models ---
class SessionBooking(BaseModel):
    user_id: str = Field(..., description="The unique identifier for the user.")
    user_name: str = Field(..., max_length=100, description="Name of the person booking the session.")
    expert_name: str = Field(..., description="Name of the expert selected for the session.")
    session_date: date = Field(..., description="The date selected for the session.")
    time_slot: time_obj = Field(..., description="The time slot selected for the session.")
    session_type: str = Field(..., description="The type of session (e.g., Yoga, Stress Relief).")
    appointment_type: str = Field(..., description="How the appointment will be conducted (e.g., Call, Video Call).")
    repeat_on: Optional[List[str]] = Field(None, description="Days of the week to repeat the session.")
    end_date: Optional[date] = Field(None, description="The end date for repeating sessions.")
    reminder: str = Field(..., description="When to send a reminder before the session.")

class SessionBookingResponse(SessionBooking):
    session_booking_id: str = Field(..., description="The unique identifier for the booked session.")
    status: str = Field("approved", description="The status of the booking.")
    google_meet_link: Optional[str] = Field(None, description="A Google Meet link, generated if the appointment is a video call.")

# CHANGED: Added `stt_duration` to accept timing from the frontend
class ConversationRequest(BaseModel):
    user_input: str
    state: dict
    stt_duration: Optional[float] = None

# --- Application Setup ---
app = FastAPI(title="LangGraph Booking API", version="8.1.0")

# --- Available Options for Booking (Context for the LLM) ---
EXPERT_NAMES = [ "Dr. Jagadesh", "Dr. Pamanji"]
SESSION_TYPES = ["Yoga", "Stress Relief", "Nutritional Advice"]
APPOINTMENT_TYPES = ["Video Call", "Audio Call"]
AVAILABLE_TIMES = [t.strftime('%I:%M %p') for t in [time_obj(h, m) for h in range(9, 18) for m in (0, 30)]]

# --- Core Service Logic ---
def _create_booking_record(booking_details: SessionBooking) -> SessionBookingResponse:
    session_id = f"session_{uuid.uuid4()}"
    response_data = booking_details.model_dump()
    if booking_details.appointment_type == "Video Call":
        meet_id = f"{uuid.uuid4().hex[:3]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:3]}"
        response_data["google_meet_link"] = f"https://meet.google.com/{meet_id}"
    return SessionBookingResponse(session_booking_id=session_id, **response_data)

async def text_to_speech_local(text: str, voice: str = "af_heart") -> Optional[bytes]:
    if not tts_pipeline:
        return None
    def run_tts_and_create_wav():
        audio_iter = tts_pipeline(text, voice=voice)
        audio_chunks = [chunk for _, _, chunk in audio_iter]
        if not audio_chunks:
            return None
        full_audio_tensor = torch.cat(audio_chunks, dim=-1)
        final_audio_numpy = full_audio_tensor.numpy()
        raw_audio_bytes = (final_audio_numpy * 32767).astype(np.int16).tobytes()
        SAMPLE_RATE = 24000
        NUM_CHANNELS = 1
        SAMPLE_WIDTH_BYTES = 2
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(NUM_CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH_BYTES)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(raw_audio_bytes)
            wav_bytes = wav_buffer.getvalue()
        return wav_bytes
    return await asyncio.to_thread(run_tts_and_create_wav)

# ========== LangGraph Agent Setup ==========
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def book_expert_session(
    user_name: str, expert_name: str, session_date: str, time_slot: str,
    session_type: str, appointment_type: str
) -> str:
    """Finalizes and books the expert session once all required information has been collected."""
    try:
        booking_request = SessionBooking(
            user_id="user_langgraph_123", user_name=user_name,
            expert_name=expert_name,
            session_date=parse_date(session_date).date(),
            time_slot=parse_date(time_slot).time(), session_type=session_type,
            appointment_type=appointment_type, reminder="15 minutes before"
        )
        booking_response = _create_booking_record(booking_request)
        return booking_response.model_dump_json()
    except (ValidationError, ValueError) as e:
        return f"Booking failed due to invalid details: {e}. Please ask the user to correct the information."

# --- LLM and Agent Definition ---
SYSTEM_PROMPT = f"""
You are a friendly and highly efficient AI assistant for booking expert sessions. Your primary goal is to collect all necessary information and then use the `book_expert_session` tool to finalize the appointment. Today's date is {date.today().strftime('%A, %B %d, %Y')}.

### Available Options:
-   **Experts**: {', '.join(EXPERT_NAMES)}
-   **Session Types**: {', '.join(SESSION_TYPES)}
-   **Appointment Types**: {', '.join(APPOINTMENT_TYPES)}
-   **Available Times**: Business hours are 9:00 AM to 5:00 PM. Guide the user to a valid time slot.

RULES:
Stick to these flow rules:Tell about availble experts, session types, and appointment types, then ask for date and time. Do not tell about available time slots unless the user asks for it.
1. Always be conversational. Start by introducing yourself. Do not ask for user_name, it will be provided from databases.
2. Ask Session Type, Date and Time in a single question. Then ask for  Appointment Type in a single question.
3. Once you have all six pieces of information, your response MUST be a final confirmation question in paragraph form, summarizing all the details and asking for confirmation to proceed with the booking. within 30 words.
4. if the user do not tell date and time Ask the date and time in a single question, e.g., "Which date and time would you like to book?"

6. If the user says "yes" or similar, respond with a final confirmation message including all booking details and end the conversation.
7. Use the current conversation history to maintain context.
8. Maintain a friendly and professional tone throughout the conversation. And use only english language.
9. Read time 11:00 AM as 11 AM, not 11:00 hours. Do not say the time till the user will ask for available slots.
10. If the user provides a date, ensure it is not in the past. If it is, politely ask for a future date.
"""

tools = [book_expert_session]
llm = ChatMistralAI(endpoint=MISTRAL_LLM_ENDPOINT, mistral_api_key=MISTRAL_API_KEY)
llm_with_tools = llm.bind_tools(tools)

def should_continue(state: AgentState) -> str:
    last_message = state['messages'][-1]
    return "tool_executor" if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0 else END

def call_llm(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

async def take_action(state: AgentState) -> dict:
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        tool_to_call = {tool.name: tool for tool in tools}[t['name']]
        result = await tool_to_call.ainvoke(t['args'])
        results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
    return {"messages": results}

agent_graph = StateGraph(AgentState)
agent_graph.add_node("llm", call_llm)
agent_graph.add_node("tool_executor", take_action)
agent_graph.add_conditional_edges("llm", should_continue, {"tool_executor": "tool_executor", END: END})
agent_graph.add_edge("tool_executor", "llm")
agent_graph.set_entry_point("llm")
assistant_agent = agent_graph.compile()

# ========== API Endpoints ==========
@app.get("/")
def read_root():
    return {"message": "LangGraph Booking API is running."}

@app.post("/book-session", response_model=SessionBookingResponse)
async def book_session_endpoint(booking_request: SessionBooking):
    return _create_booking_record(booking_request)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    if not stt_model:
        raise HTTPException(status_code=500, detail="STT model is not initialized.")
    try:
        temp_filename = f"temp_audio_{uuid.uuid4()}.wav"
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())
        
        stt_start_time = time.perf_counter()
        segments, _ = await asyncio.to_thread(stt_model.transcribe, temp_filename, language="en")
        transcription = " ".join([segment.text for segment in segments])
        stt_end_time = time.perf_counter()
        stt_duration = stt_end_time - stt_start_time
        
        os.remove(temp_filename)
        
        print("\n" + "="*50)
        print("🎙️  Speech-to-Text (STT) Processing")
        print("="*50)
        print(f"Transcribed Text: {transcription.strip()}")
        print(f"Time Taken (STT): {stt_duration:.2f} seconds")
        print("="*50 + "\n")
        
        # CHANGED: Return the duration along with the text
        return JSONResponse(content={
            "transcribed_text": transcription.strip() or "Could not understand audio.",
            "stt_duration": stt_duration
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during transcription: {e}")

@app.post("/tts")
async def text_to_speech_endpoint(request: TTSRequest):
    """
    Endpoint to convert text to speech and return the audio file.
    """
    if not tts_pipeline:
        raise HTTPException(status_code=500, detail="TTS model is not initialized.")
    
    try:
        # Use the existing helper function to generate audio
        audio_content = await text_to_speech_local(request.text, voice=request.voice)

        if audio_content is None:
            raise HTTPException(status_code=500, detail="Failed to generate audio.")

        # Return the raw audio bytes with the correct content type
        return Response(content=audio_content, media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during TTS processing: {e}")
    
    
@app.post("/converse-booking-stream")
async def converse_booking_stream_endpoint(request: ConversationRequest):
    async def event_stream() -> AsyncGenerator[bytes, None]:
        # CHANGED: Start a timer for the total request-to-response duration
        total_start_time = time.perf_counter()

        print("="*50)
        print(f"👤 User Message: {request.user_input}")
        print("="*50)

        conversation_history = [
            AIMessage(content=msg['content']) if msg['role'] == 'assistant' else HumanMessage(content=msg['content'])
            for msg in request.state.get('conversation_history', [])
        ]
        if request.user_input:
            conversation_history.append(HumanMessage(content=request.user_input))
        initial_state = {"messages": [SystemMessage(content=SYSTEM_PROMPT)] + conversation_history}
        full_response_text = ""
        final_booking_details = None
        updated_state = request.state.copy()
        
        llm_start_time = time.perf_counter()
        final_state = await assistant_agent.ainvoke(initial_state)
        llm_end_time = time.perf_counter()
        llm_duration = llm_end_time - llm_start_time
        
        last_message = final_state['messages'][-1]
        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
            full_response_text = last_message.content
        if len(final_state['messages']) > 2 and isinstance(final_state['messages'][-2], ToolMessage):
            tool_message = final_state['messages'][-2]
            try:
                booking_data = json.loads(tool_message.content)
                final_booking_details = booking_data
                full_response_text = final_state['messages'][-1].content
            except (json.JSONDecodeError, TypeError):
                 print(f"Tool returned non-JSON content: {tool_message.content}")
                 full_response_text = "There was an issue processing the booking."

        print(f"🤖 AI Message: {full_response_text}")

        if 'conversation_history' not in updated_state:
            updated_state['conversation_history'] = []
        if request.user_input:
            updated_state['conversation_history'].append({"role": "user", "content": request.user_input})
        if full_response_text:
            updated_state['conversation_history'].append({"role": "assistant", "content": full_response_text})
        
        tts_start_time = time.perf_counter()
        audio_content = await text_to_speech_local(full_response_text)
        tts_end_time = time.perf_counter()
        tts_duration = tts_end_time - tts_start_time

        # CHANGED: Calculate total duration and print the full summary
        total_end_time = time.perf_counter()
        total_duration = total_end_time - total_start_time

        print("--- Performance Metrics ---")
        if request.stt_duration is not None:
            print(f"Time Taken (STT): {request.stt_duration:.2f} seconds")
        print(f"Time Taken (LLM): {llm_duration:.2f} seconds")
        print(f"Time Taken (TTS): {tts_duration:.2f} seconds")
        print("-" * 27)
        print(f"Total Backend Time: {total_duration:.2f} seconds")
        print("="*50 + "\n")


        if final_booking_details:
            updated_state['booking_complete'] = True
            confirm_payload = {
                "type": "final_confirmation", "text": full_response_text,
                "audio": audio_content.hex() if audio_content else None,
                "booking_details": json.dumps(final_booking_details),
                "updated_state": json.dumps(updated_state)
            }
            yield (json.dumps(confirm_payload) + '\n').encode('utf-8')
        else:
            if full_response_text:
                chunk_payload = {"type": "text_delta", "content": full_response_text}
                yield (json.dumps(chunk_payload) + '\n').encode('utf-8')
            if audio_content:
                audio_payload = {"type": "full_audio", "content": audio_content.hex()}
                yield (json.dumps(audio_payload) + '\n').encode('utf-8')
            state_payload = {"type": "state_update", "content": json.dumps(updated_state)}
            yield (json.dumps(state_payload) + '\n').encode('utf-8')

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")