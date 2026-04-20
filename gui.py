import cv2
import mediapipe as mp
import numpy as np
import joblib
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import pyttsx3
import threading
from datetime import datetime

# --- NEW IMPORT FOR STEP B ---
from data_manager.database import log_detection 

# -------------------------
# Load Model
# -------------------------
model = joblib.load("model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = None
running = False

# -------------------------
# Text-to-Speech Setup
# -------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
last_spoken = ""

def speak_async(text):
    t = threading.Thread(target=lambda: speak(text))
    t.start()

def speak(text):
    global last_spoken
    if text != last_spoken:
        last_spoken = text
        engine.say(text)
        engine.runAndWait()

# -------------------------
# GUI Setup
# -------------------------
root = Tk()
root.title("SignVision – Hand Sign Detector & Analytics")
root.geometry("1100x700") # Widened to fit analytics sidebar
root.configure(bg="#1e1e1e")

# Main Container for Layout
main_container = Frame(root, bg="#1e1e1e")
main_container.pack(fill=BOTH, expand=True)

# LEFT SIDE: Video and Controls
left_frame = Frame(main_container, bg="#1e1e1e")
left_frame.pack(side=LEFT, padx=20, pady=10, fill=BOTH, expand=True)

title_lbl = Label(left_frame, text="SignVision Detector", font=("Arial", 22, "bold"), fg="#00ffcc", bg="#1e1e1e")
title_lbl.pack(pady=5)

video_frame = Label(left_frame, bg="#000000")
video_frame.pack(pady=10)

prediction_lbl = Label(left_frame, text="Show a Sign", font=("Arial", 20, "bold"), fg="#00ff00", bg="#1e1e1e")
prediction_lbl.pack(pady=10)

# RIGHT SIDE: Step 3 - Analytics Sidebar
right_frame = Frame(main_container, bg="#2d2d2d", width=250)
right_frame.pack(side=RIGHT, fill=Y, padx=10, pady=10)

stats_title = Label(right_frame, text="Live Analytics", font=("Arial", 16, "bold"), fg="white", bg="#2d2d2d")
stats_title.pack(pady=15)

# Session counters
session_count = 0
session_lbl = Label(right_frame, text=f"Total Signs: {session_count}", font=("Arial", 12), fg="#00ffcc", bg="#2d2d2d")
session_lbl.pack(pady=5)

recent_history_lbl = Label(right_frame, text="Recent Detection:", font=("Arial", 10, "italic"), fg="#aaaaaa", bg="#2d2d2d")
recent_history_lbl.pack(pady=(20, 0))

history_box = Text(right_frame, height=15, width=25, bg="#1e1e1e", fg="white", font=("Courier", 10), state=DISABLED)
history_box.pack(padx=10, pady=5)

# -------------------------
# Analytics Update Function
# -------------------------
def update_analytics_ui(pred):
    global session_count
    session_count += 1
    session_lbl.config(text=f"Total Signs: {session_count}")
    
    # Log to Sidebar History
    history_box.config(state=NORMAL)
    timestamp = datetime.now().strftime("%H:%M:%S")
    history_box.insert(END, f"[{timestamp}] -> {pred}\n")
    history_box.see(END) # Auto-scroll
    history_box.config(state=DISABLED)

# -------------------------
# Detection Logic
# -------------------------
def start_detection():
    global cap, running
    if running: return
    cap = cv2.VideoCapture(0)
    running = True
    update_frame()

def stop_detection():
    global running, last_spoken
    running = False
    last_spoken = ""
    prediction_lbl.config(text="Detection Stopped", fg="red")

def update_frame():
    global running, cap

    if running:
        ret, frame = cap.read()
        if not ret: return
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        pred_text = "No Hand Detected"

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                
                xs = [lm.x for lm in hand.landmark]
                ys = [lm.y for lm in hand.landmark]
                row = np.array(xs + ys).reshape(1, -1)

                pred = str(model.predict(row)[0])
                pred_text = f"Detected Sign: {pred}"

                # ---- STEP B & STEP 3 INTEGRATION ----
                if pred != last_spoken: # Only log/speak when sign changes
                    log_detection(pred)       # Push to SQLite
                    update_analytics_ui(pred) # Push to Screen
                    speak_async(pred)         # TTS

        prediction_lbl.config(text=pred_text, fg="#00ff00")

        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_frame.imgtk = imgtk
        video_frame.configure(image=imgtk)

    video_frame.after(10, update_frame)

# -------------------------
# Footer Buttons
# -------------------------
btn_frame = Frame(left_frame, bg="#1e1e1e")
btn_frame.pack(pady=10)

start_btn = Button(btn_frame, text="START", command=start_detection, font=("Arial", 14, "bold"), fg="white", bg="#0066ff", width=12)
start_btn.grid(row=0, column=0, padx=10)

stop_btn = Button(btn_frame, text="STOP", command=stop_detection, font=("Arial", 14, "bold"), fg="white", bg="#ff0000", width=12)
stop_btn.grid(row=0, column=1, padx=10)

def on_close():
    global cap, running
    running = False
    if cap: cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()