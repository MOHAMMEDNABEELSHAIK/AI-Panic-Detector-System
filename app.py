import os
os.environ["HF_HOME"] = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"

import tkinter as tk
from tkinter import messagebox
import threading
import queue
import json
import sqlite3
import time
import random
import math
import re
import collections

import numpy as np
import sounddevice as sd

from vosk import Model, KaldiRecognizer
from transformers import pipeline
from twilio.rest import Client
import geocoder

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

# =========================================================
# DESIGN TOKENS
# =========================================================

BG         = "#07090f"
SURFACE    = "#0d1117"
SURFACE2   = "#111827"
BORDER     = "#1f2937"
BORDER_LIT = "#2d3f55"
GHOST      = "#0a1020"

CYAN       = "#00e5ff"
CYAN_DIM   = "#0099bb"
GREEN      = "#00ffa3"
GREEN_DIM  = "#00b37a"
RED        = "#ff3355"
RED_DIM    = "#cc2244"
AMBER      = "#ffb300"
WHITE      = "#f0f6fc"
MUTED      = "#4a5568"

# =========================================================
# ROOT WINDOW
# =========================================================

root = tk.Tk()
root.title("Panic Detector")
root.geometry("1200x820")
root.configure(bg=BG)
root.minsize(1000, 700)

# =========================================================
# DATABASE
# =========================================================

conn   = sqlite3.connect("panic_detector.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS contacts(
        id    INTEGER PRIMARY KEY AUTOINCREMENT,
        name  TEXT NOT NULL,
        phone TEXT NOT NULL
    )
""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS alert_log(
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        trigger   TEXT,
        score     INTEGER,
        sent      INTEGER DEFAULT 0
    )
""")
conn.commit()

# =========================================================
# PANIC KEYWORD SYSTEM — THREE TIERS
# =========================================================

# ── TIER 1: Single words that alone mean emergency ────────
# FIX: "help" IS now here — it's unambiguous when spoken alone
# The repetition tracker handles false positives (e.g. "help yourself")
CRITICAL_WORDS = {
    # English
    "emergency", "danger", "attack",
    "rape", "murder", "kidnap", "kidnapped", "bleeding",
    "dying", "hostage", "gunshot", "stabbed", "mayday", "sos",
    "bomb", "shooting", "intruder",
    # FIX: "help" alone triggers — repetition guard prevents false positives
    "help",
    # Hindi / Hinglish
    "bachao", "madad", "chhodo", "choro",
    # Tamil
    "kapathunga", "rakshikkanam", "udavi",
    # Telugu
    "aagipo", "sahayam",
}

# ── TIER 2: Panic phrases — score +6 each ─────────────────
PANIC_PHRASES = {
    "help me", "help me please", "please help me",
    "save me", "please help", "someone help", "anybody help",
    "call police", "call 911", "call 100", "call 112", "call ambulance",
    "let me go", "let me out", "leave me alone", "get away from me",
    "don't touch me", "stop touching", "he is hurting me",
    "she is hurting me", "they are hurting me",
    "i am being attacked", "i am in danger", "i need help",
    "i am scared", "i am bleeding", "i can't breathe",
    "i cannot breathe", "can't breathe",
    "help help", "please stop", "get off me", "leave me",
    "i'm scared", "i'm hurt", "i'm bleeding",
    "someone is following me", "i am being followed",
    "he has a knife", "he has a gun", "there's a gun",
    "i am trapped", "i am stuck", "i am lost",
    # Hindi
    "mujhe bachao", "madad karo", "police bulao",
    "chhod do", "mat maro", "mujhe jaane do",
    "koi bachao", "police ko bulao",
    # Tamil
    "enakku udavi", "police ku sollunga", "vidunga",
    "yaaravadu help pannunga",
    # Telugu
    "naaku sahaayam", "police ki cheppandi",
}

# ── TIER 3: Support words — score +2 each ─────────────────
SUPPORT_WORDS = {
    "hurt", "pain", "scared", "afraid", "terrified",
    "threatened", "trapped", "unconscious", "injured",
    "wounded", "suffocating", "choking", "drowning",
    "burning", "smoke", "screaming", "screamed",
    "stranger", "robber", "thief", "weapon",
    "knife", "gun", "pistol", "shot", "blood",
    "accident", "crash", "collapsed", "fell", "stuck",
    "locked", "broken", "lost", "missing", "followed",
    "following", "chasing", "chase",
    # Hindi support
    "dard", "darr", "khatra", "chot", "khoon",
    # Tamil support
    "valikuthu", "bayam", "aapaththu", "ratham",
}

STOPWORDS = {
    "the","and","you","that","this","with","from","have",
    "are","was","were","will","would","could","should",
    "what","when","where","which","who","how","all",
    "not","but","for","can","get","got","its","your",
    "our","been","they","them","their","then","than",
    "good","idea","yeah","okay","yes","well","just",
    "like","know","think","going","come","came","see",
    "look","make","take","tell","said","some","more",
    "out","about","into","over","also","very","too",
    "bear","mark","data","love","laugh","figure","bet",
    "oh","um","uh","hmm","okay","hey","hi","hello",
}

ALERT_THRESHOLD = 6

# ── Load dataset ──────────────────────────────────────────
DATASET_LOADED  = False
DATASET_PHRASES = 0

if PANDAS_OK:
    try:
        dataset = pd.read_csv("panic_dataset_multilingual_20k.csv")
        dataset.columns = dataset.columns.str.lower()
        text_col  = "text"  if "text"  in dataset.columns else dataset.columns[0]
        label_col = "label" if "label" in dataset.columns else None
        dataset[text_col] = dataset[text_col].astype(str).str.lower().str.strip()

        added = 0
        for _, row in dataset.iterrows():
            sentence = row[text_col]
            if label_col:
                lbl = str(row[label_col]).strip().lower()
                is_panic = lbl in ("1","panic","true","yes","danger","emergency")
            else:
                is_panic = True
            if is_panic:
                words    = sentence.split()
                non_stop = [w for w in words if w not in STOPWORDS]
                # FIX: Only add ASCII-only phrases (so Vosk English STT can match them)
                # Non-ASCII multilingual phrases are already in PANIC_PHRASES above
                if (2 <= len(words) <= 6
                        and len(non_stop) >= 1
                        and sentence.isascii()):
                    PANIC_PHRASES.add(sentence)
                    added += 1

        DATASET_LOADED  = True
        DATASET_PHRASES = added
        print(f"✅ Dataset loaded — {added} phrases added  |  total: {len(PANIC_PHRASES)}")
    except FileNotFoundError:
        print("⚠  Dataset CSV not found — using built-in keywords only")
    except Exception as e:
        print(f"⚠  Dataset error: {e}")

# =========================================================
# TWILIO CONFIG  ← Replace with your credentials
# =========================================================

ACCOUNT_SID = "YOUR_TWILIO_ACCOUNT_SID"
AUTH_TOKEN = "YOUR_TWILIO_AUTH_TOKEN"
TWILIO_NUMBER = "YOUR_TWILIO_NUMBER"

try:
    twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)
    TWILIO_OK = True
except Exception as e:
    twilio_client = None
    TWILIO_OK = False
    print(f"Twilio init error: {e}")

VERIFIED_CONTACTS = []

def fetch_verified_numbers():
    global VERIFIED_CONTACTS
    if not TWILIO_OK:
        root.after(0, lambda: _refresh_verified_panel(error="Twilio not configured"))
        return
    try:
        records = twilio_client.outgoing_caller_ids.list()
        VERIFIED_CONTACTS = [
            {"name": r.friendly_name or r.phone_number, "phone": r.phone_number}
            for r in records
        ]
        print(f"✅ Verified numbers: {[c['phone'] for c in VERIFIED_CONTACTS]}")
        root.after(0, _refresh_verified_panel)
    except Exception as e:
        print("Twilio fetch error:", e)
        root.after(0, lambda: _refresh_verified_panel(error=str(e)))

# =========================================================
# NLP MODEL
# =========================================================

emotion_classifier = None
nlp_loaded         = False

def load_nlp_model():
    global emotion_classifier, nlp_loaded
    try:
        print("Loading NLP model…")
        emotion_classifier = pipeline(
            task="text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            top_k=None
        )
        nlp_loaded = True
        print("✅ NLP model loaded")
        root.after(0, lambda: _set_nlp_badge(True))
    except Exception as e:
        print("NLP load error:", e)
        root.after(0, lambda: _set_nlp_badge(False))

# =========================================================
# VOSK
# =========================================================

print("Loading Vosk speech model…")
try:
    vosk_model = Model("vosk-model-small-en-us-0.15")
    print("✅ Vosk loaded")
except Exception:
    messagebox.showerror("Vosk Error",
        "Cannot find vosk-model-small-en-us-0.15/\n"
        "Download from: https://alphacephei.com/vosk/models")
    root.destroy()

recognizer = KaldiRecognizer(vosk_model, 16000)
recognizer.SetWords(True)

# =========================================================
# GLOBALS
# =========================================================

running         = True
audio_queue     = queue.Queue()
audio_stream    = None
current_volume  = 0
last_alert_time = 0
ALERT_COOLDOWN  = 20
system_state    = "monitoring"
alert_count     = [0]

# ── FIX: Repetition tracker ───────────────────────────────
# Tracks panic words/phrases heard in a rolling 15-second window.
# If the same critical word appears 2+ times → instant trigger.
# This fixes: user says "help" once (no trigger), says it again → trigger.
_recent_words   = collections.deque()   # (timestamp, word)
REPEAT_WINDOW   = 15.0                  # seconds
REPEAT_THRESHOLD = 2                    # same word N times = panic

def _record_word_repetition(word):
    """
    Add word to rolling window. Return True if this word has been
    seen REPEAT_THRESHOLD or more times within REPEAT_WINDOW seconds.
    """
    now = time.time()
    # Prune old entries
    while _recent_words and now - _recent_words[0][0] > REPEAT_WINDOW:
        _recent_words.popleft()
    _recent_words.append((now, word))
    count = sum(1 for _, w in _recent_words if w == word)
    print(f"  [repeat] '{word}' × {count} in last {REPEAT_WINDOW}s")
    return count >= REPEAT_THRESHOLD

# =========================================================
# UI — TOP BAR
# =========================================================

topbar = tk.Frame(root, bg=SURFACE, height=66)
topbar.pack(fill="x")
topbar.pack_propagate(False)

tk.Label(topbar, text="🛡", font=("Segoe UI Emoji", 28),
         fg=CYAN, bg=SURFACE).pack(side="left", padx=(20, 6), pady=10)

tk.Label(topbar, text="AI PANIC DETECTOR",
         font=("Courier New", 18, "bold"),
         fg=WHITE, bg=SURFACE).pack(side="left", pady=10)

tk.Label(topbar, text="v2.1",
         font=("Courier New", 9),
         fg=MUTED, bg=SURFACE).pack(side="left", padx=(8, 0), pady=10, anchor="s")

def _badge(text, fg, bg):
    return tk.Label(topbar, text=text,
                    font=("Courier New", 9, "bold"),
                    fg=fg, bg=bg, padx=10, pady=4)

dataset_badge = _badge("  CSV: CHECKING  ", AMBER, GHOST)
dataset_badge.pack(side="right", padx=4, pady=18)

nlp_badge = _badge("  NLP: LOADING  ", AMBER, GHOST)
nlp_badge.pack(side="right", padx=4, pady=18)

def _set_nlp_badge(loaded):
    if loaded:
        nlp_badge.config(text="  NLP: READY  ", fg=GREEN, bg="#0a1f12")
    else:
        nlp_badge.config(text="  NLP: OFFLINE  ", fg=AMBER, bg="#1f1505")

tk.Frame(root, bg=CYAN, height=2).pack(fill="x")

# =========================================================
# UI — BODY
# =========================================================

body = tk.Frame(root, bg=BG)
body.pack(fill="both", expand=True, padx=16, pady=12)

left_col = tk.Frame(body, bg=BG)
left_col.pack(side="left", fill="both", expand=True, padx=(0, 12))

right_col = tk.Frame(body, bg=BG, width=310)
right_col.pack(side="right", fill="y")
right_col.pack_propagate(False)

# ── Status cards ──────────────────────────────────────────

def _card(parent, top_label, value, color):
    frame = tk.Frame(parent, bg=SURFACE2,
                     highlightbackground=BORDER, highlightthickness=1)
    frame.pack(side="left", fill="both", expand=True, padx=4)
    tk.Label(frame, text=top_label,
             font=("Courier New", 8), fg=MUTED, bg=SURFACE2
             ).pack(anchor="w", padx=12, pady=(10, 2))
    lbl = tk.Label(frame, text=value,
                   font=("Courier New", 13, "bold"),
                   fg=color, bg=SURFACE2)
    lbl.pack(anchor="w", padx=12, pady=(0, 10))
    return lbl

status_row = tk.Frame(left_col, bg=BG)
status_row.pack(fill="x", pady=(0, 10))

lbl_status  = _card(status_row, "SYSTEM STATE",  "● MONITORING",  GREEN)
lbl_mic     = _card(status_row, "MICROPHONE",    "● ACTIVE",      CYAN)
lbl_alerts  = _card(status_row, "ALERTS SENT",   "0",             WHITE)
lbl_phrases = _card(status_row, "PANIC PHRASES", str(len(PANIC_PHRASES)), AMBER)

# ── Waveform ──────────────────────────────────────────────

wave_frame = tk.Frame(left_col, bg=SURFACE2,
                      highlightbackground=BORDER, highlightthickness=1)
wave_frame.pack(fill="x", pady=(0, 10))

tk.Label(wave_frame, text="LIVE AUDIO",
         font=("Courier New", 8, "bold"),
         fg=MUTED, bg=SURFACE2).pack(anchor="w", padx=14, pady=(10, 0))

canvas = tk.Canvas(wave_frame, height=100, bg=SURFACE2, highlightthickness=0)
canvas.pack(fill="x", padx=14, pady=(2, 12))

BAR_COUNT  = 80
bar_ids    = []
_wave_tick = [0]

def _build_bars():
    canvas.update_idletasks()
    W   = canvas.winfo_width() or 700
    gap = W / BAR_COUNT
    bar_ids.clear()
    canvas.delete("all")
    for i in range(BAR_COUNT):
        x = i * gap + gap / 2
        b = canvas.create_rectangle(x-2, 50, x+2, 50, fill=CYAN, outline="")
        bar_ids.append(b)

canvas.after(300, _build_bars)

def update_visualizer(volume):
    _wave_tick[0] += 1
    t   = _wave_tick[0]
    W   = canvas.winfo_width() or 700
    gap = W / BAR_COUNT
    level = max(3, min(int(volume), 48))
    for i, bar in enumerate(bar_ids):
        wave  = math.sin((i / BAR_COUNT) * math.pi * 4 + t * 0.2) * (level * 0.4)
        noise = random.randint(-level // 4, level // 4)
        h     = max(2, level + int(wave) + noise)
        cy    = 50
        x     = i * gap + gap / 2
        canvas.coords(bar, x-2, cy - h, x+2, cy + h)
        col = RED if h > 34 else AMBER if h > 20 else CYAN
        canvas.itemconfig(bar, fill=col)

# ── Transcript log ────────────────────────────────────────

log_frame = tk.Frame(left_col, bg=SURFACE2,
                     highlightbackground=BORDER, highlightthickness=1)
log_frame.pack(fill="both", expand=True)

log_header = tk.Frame(log_frame, bg=SURFACE2)
log_header.pack(fill="x", padx=14, pady=(10, 2))

tk.Label(log_header, text="SPEECH TRANSCRIPT",
         font=("Courier New", 8, "bold"),
         fg=MUTED, bg=SURFACE2).pack(side="left")

live_dot = tk.Label(log_header, text="● LIVE",
                    font=("Courier New", 8), fg=RED, bg=SURFACE2)
live_dot.pack(side="right")

log_box = tk.Text(log_frame, bg=SURFACE2, fg=WHITE,
                  font=("Courier New", 10),
                  state="disabled", relief="flat", bd=0,
                  insertbackground=CYAN,
                  selectbackground=BORDER_LIT,
                  wrap="word", highlightthickness=0,
                  spacing1=3, spacing3=3)
log_box.pack(fill="both", expand=True, padx=14, pady=(0, 12))

log_box.tag_config("ts",     foreground=MUTED)
log_box.tag_config("normal", foreground=CYAN_DIM)
log_box.tag_config("panic",  foreground=RED)
log_box.tag_config("score",  foreground=AMBER)
log_box.tag_config("info",   foreground=GREEN_DIM)
log_box.tag_config("repeat", foreground=AMBER)   # FIX: new tag for repetition warnings

def _log(text, tag="normal", score=None):
    ts = time.strftime("%H:%M:%S")
    log_box.config(state="normal")
    log_box.insert("end", f"[{ts}] ", "ts")
    if score is not None:
        log_box.insert("end", f"[{score:>2}] ", "score")
    log_box.insert("end", text + "\n", tag)
    log_box.see("end")
    log_box.config(state="disabled")

# ── Right column helpers ───────────────────────────────────

def _section(parent, title, pady=(0, 10)):
    f = tk.Frame(parent, bg=SURFACE2,
                 highlightbackground=BORDER, highlightthickness=1)
    f.pack(fill="x", pady=pady)
    tk.Label(f, text=title,
             font=("Courier New", 8, "bold"),
             fg=MUTED, bg=SURFACE2).pack(anchor="w", padx=14, pady=(10, 4))
    return f

# ── Alert console ─────────────────────────────────────────
alert_sec = _section(right_col, "ALERT CONSOLE")

pulse_canvas = tk.Canvas(alert_sec, width=290, height=34,
                          bg=SURFACE2, highlightthickness=0)
pulse_canvas.pack(padx=14, pady=(0, 2))
_pdot = pulse_canvas.create_oval(4, 8, 22, 26, fill=GREEN, outline="")
_ptxt = pulse_canvas.create_text(30, 17, text="AI Monitoring Active",
                                  fill=GREEN, anchor="w",
                                  font=("Courier New", 11, "bold"))
_pulse_phase = [0]

def _animate_pulse():
    _pulse_phase[0] += 1
    cols = [RED, RED_DIM] if system_state == "alert" else [GREEN, GREEN_DIM]
    col  = cols[_pulse_phase[0] % 2]
    pulse_canvas.itemconfig(_pdot, fill=col)
    root.after(500, _animate_pulse)

root.after(500, _animate_pulse)

alert_detail = tk.Label(alert_sec, text="Listening for panic keywords…",
                         font=("Courier New", 9), fg=MUTED, bg=SURFACE2,
                         wraplength=260, justify="left")
alert_detail.pack(anchor="w", padx=14, pady=(0, 12))

# ── Panic score bar ───────────────────────────────────────
score_sec = _section(right_col, "PANIC SCORE")

score_bar_bg   = tk.Canvas(score_sec, height=14, bg=GHOST, highlightthickness=0)
score_bar_bg.pack(fill="x", padx=14, pady=(0, 4))
score_bar_fill = score_bar_bg.create_rectangle(0, 0, 0, 14, fill=GREEN, outline="")

score_row = tk.Frame(score_sec, bg=SURFACE2)
score_row.pack(fill="x", padx=14, pady=(0, 4))
score_label = tk.Label(score_row, text="0 / 10",
                        font=("Courier New", 12, "bold"),
                        fg=WHITE, bg=SURFACE2)
score_label.pack(side="left")
score_tier  = tk.Label(score_row, text="SAFE",
                        font=("Courier New", 9),
                        fg=GREEN, bg=SURFACE2)
score_tier.pack(side="right")

threshold_label = tk.Label(score_sec,
    text=f"Threshold: {ALERT_THRESHOLD}  |  Cooldown: {ALERT_COOLDOWN}s  |  Repeat×{REPEAT_THRESHOLD} in {REPEAT_WINDOW:.0f}s",
    font=("Courier New", 8), fg=MUTED, bg=SURFACE2)
threshold_label.pack(anchor="w", padx=14, pady=(0, 10))

def _update_score_bar(score):
    score_bar_bg.update_idletasks()
    W    = score_bar_bg.winfo_width() or 270
    frac = min(score / 10.0, 1.0)
    col  = RED if frac >= 0.6 else AMBER if frac >= 0.35 else GREEN
    score_bar_bg.coords(score_bar_fill, 0, 0, int(W * frac), 14)
    score_bar_bg.itemconfig(score_bar_fill, fill=col)
    tier = "DANGER" if frac >= 0.6 else "CAUTION" if frac >= 0.35 else "SAFE"
    score_label.config(text=f"{score} / 10", fg=col)
    score_tier.config(text=tier, fg=col)

# ── Action buttons ────────────────────────────────────────
actions_sec = _section(right_col, "ACTIONS")

def _btn(parent, text, bg_col, cmd):
    def _lighter(c):
        r = min(255, int(c[1:3], 16) + 30)
        g = min(255, int(c[3:5], 16) + 30)
        b = min(255, int(c[5:7], 16) + 30)
        return f"#{r:02x}{g:02x}{b:02x}"
    frm = tk.Frame(parent, bg=bg_col, cursor="hand2")
    frm.pack(fill="x", padx=14, pady=3)
    lbl = tk.Label(frm, text=text,
                   font=("Courier New", 10, "bold"),
                   fg=BG if bg_col not in (SURFACE2,) else WHITE,
                   bg=bg_col, pady=8)
    lbl.pack(fill="x")
    for w in (frm, lbl):
        w.bind("<Button-1>", lambda e: cmd())
        w.bind("<Enter>",    lambda e,f=frm,l=lbl,c=_lighter(bg_col): (f.config(bg=c), l.config(bg=c)))
        w.bind("<Leave>",    lambda e,f=frm,l=lbl: (f.config(bg=bg_col), l.config(bg=bg_col)))
    return frm

_btn(actions_sec, "🚨  MANUAL PANIC ALERT",  RED,      lambda: trigger_panic("Manual trigger", score=10))
_btn(actions_sec, "🔄  REFRESH CONTACTS",    CYAN,     lambda: threading.Thread(target=fetch_verified_numbers, daemon=True).start())
_btn(actions_sec, "👤  MANAGE CONTACTS",     SURFACE2, lambda: _open_manage_contacts())
_btn(actions_sec, "📋  ALERT HISTORY",       SURFACE2, lambda: _open_alert_history())
tk.Frame(actions_sec, bg=BG, height=8).pack()

# ── Verified numbers ──────────────────────────────────────
verified_sec = _section(right_col, "TWILIO VERIFIED NUMBERS", pady=(0, 4))

_vh = tk.Frame(verified_sec, bg=SURFACE2)
_vh.pack(fill="x", padx=14, pady=(0, 2))
tk.Label(_vh, text="", font=("Courier New", 8), fg=MUTED, bg=SURFACE2).pack(side="left")
verified_dot = tk.Label(_vh, text="●", font=("Courier New", 9), fg=AMBER, bg=SURFACE2)
verified_dot.pack(side="right")

verified_list_frame = tk.Frame(verified_sec, bg=SURFACE2)
verified_list_frame.pack(fill="x", padx=14, pady=(0, 10))

tk.Label(verified_list_frame, text="Fetching…",
         font=("Courier New", 9), fg=AMBER, bg=SURFACE2).pack(anchor="w")

def _refresh_verified_panel(error=None):
    for w in verified_list_frame.winfo_children():
        w.destroy()
    if error:
        verified_dot.config(fg=RED)
        tk.Label(verified_list_frame, text=f"⚠ {error[:55]}",
                 font=("Courier New", 8), fg=RED, bg=SURFACE2,
                 wraplength=250, justify="left").pack(anchor="w", pady=4)
        return
    if not VERIFIED_CONTACTS:
        verified_dot.config(fg=AMBER)
        tk.Label(verified_list_frame,
                 text="No verified numbers.\nAdd at twilio.com → Verified Caller IDs",
                 font=("Courier New", 8), fg=MUTED, bg=SURFACE2,
                 wraplength=250, justify="left").pack(anchor="w", pady=4)
        return
    verified_dot.config(fg=GREEN)
    for c in VERIFIED_CONTACTS:
        row = tk.Frame(verified_list_frame, bg=GHOST,
                       highlightbackground=BORDER, highlightthickness=1)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="✔", font=("Courier New", 9, "bold"),
                 fg=GREEN, bg=GHOST).pack(side="left", padx=(8, 4), pady=5)
        tk.Label(row, text=c["phone"], font=("Courier New", 9, "bold"),
                 fg=WHITE, bg=GHOST).pack(side="left", pady=5)
        tk.Label(row, text=c["name"], font=("Courier New", 8),
                 fg=MUTED, bg=GHOST).pack(side="right", padx=8, pady=5)

# ── Status bar ────────────────────────────────────────────
tk.Frame(root, bg=BORDER, height=1).pack(fill="x")
statusbar = tk.Frame(root, bg=SURFACE, height=28)
statusbar.pack(fill="x")
statusbar.pack_propagate(False)

statusbar_lbl = tk.Label(statusbar, text="● SYSTEM ONLINE  |  MICROPHONE ACTIVE",
                          font=("Courier New", 8), fg=GREEN, bg=SURFACE)
statusbar_lbl.pack(side="left", padx=14, pady=6)

clock_lbl = tk.Label(statusbar, font=("Courier New", 8), fg=MUTED, bg=SURFACE)
clock_lbl.pack(side="right", padx=14)

def _tick():
    clock_lbl.config(text=time.strftime("  %Y-%m-%d  %H:%M:%S  "))
    root.after(1000, _tick)
_tick()

# =========================================================
# STATE MANAGEMENT
# =========================================================

def _set_monitoring():
    global system_state
    system_state = "monitoring"
    lbl_status.config(text="● MONITORING", fg=GREEN)
    pulse_canvas.itemconfig(_ptxt, text="AI Monitoring Active", fill=GREEN)
    alert_detail.config(text="Listening for panic keywords…", fg=MUTED)

def _set_alert():
    global system_state
    system_state = "alert"
    lbl_status.config(text="⚠ PANIC DETECTED", fg=RED)
    pulse_canvas.itemconfig(_ptxt, text="PANIC DETECTED!", fill=RED)

# =========================================================
# TEXT NORMALIZER
# =========================================================

def _normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

# =========================================================
# IMPROVED PHRASE MATCHING
# =========================================================

def _phrase_score(text, spoken_words):
    """
    Returns (score, reasons).
    FIX: word-set match now gives +6 (same as substring), not +4.
    FIX: also matches strip of filler words (uh, hey, bro) from start/end.
    """
    score   = 0
    reasons = []

    # Strip common filler from start/end for cleaner matching
    filler  = {"uh", "hey", "bro", "please", "now", "um", "hmm"}
    clean_words = [w for w in text.split() if w not in filler]
    clean_text  = " ".join(clean_words)

    for phrase in PANIC_PHRASES:
        # 1. Exact substring in full text
        if phrase in text:
            score += 6
            reasons.append(f"phrase:'{phrase}'")
            continue
        # 2. Exact substring in filler-stripped text
        if phrase in clean_text:
            score += 6
            reasons.append(f"phrase(clean):'{phrase}'")
            continue
        # 3. All words of phrase present anywhere in utterance
        phrase_words = set(phrase.split())
        if phrase_words and phrase_words.issubset(spoken_words):
            score += 6   # FIX: was 4, now 6 — equals substring credit
            reasons.append(f"word-set:'{phrase}'")

    return score, reasons

# =========================================================
# CORE ANALYZE FUNCTION
# =========================================================

def analyze_text(text):
    """
    Called once per full Vosk utterance.
    FIX: Critical single words (including "help") now also go through
         the repetition tracker so saying "help" once = log but no alert,
         saying "help" twice within 15 s = instant alert.
    """
    text  = _normalize(text)
    words = set(text.split())

    if len(text) < 2:
        return

    score   = 0
    reasons = []

    # ── TIER 1: Critical words — check repetition ─────────
    critical_hits = words & CRITICAL_WORDS
    if critical_hits:
        for word in critical_hits:
            repeat_triggered = _record_word_repetition(word)
            if repeat_triggered:
                # Heard this word 2+ times → instant alert
                score = 10
                reasons.append(f"REPEAT-CRITICAL:'{word}'×{REPEAT_THRESHOLD}")
                root.after(0, lambda: _update_score_bar(10))
                root.after(0, lambda w=word: (
                    _log(f"🔁 REPEAT PANIC word: '{w}' — ALERTING", tag="panic"),
                ))
                root.after(0, lambda: trigger_panic(text, score=10, reasons=reasons))
                return
            else:
                # First occurrence of critical word — raise score but no instant trigger
                # (avoids false positives like "help yourself")
                score += 5
                reasons.append(f"critical(1st):'{word}'")
                root.after(0, lambda w=word: (
                    _log(f"⚠ Panic word heard: '{w}' — say again to trigger alert", tag="repeat"),
                    alert_detail.config(
                        text=f"⚠ Heard '{w}' — say it again to alert!", fg=AMBER),
                    _update_score_bar(5)
                ))

    # If a single critical word already pushes us to 5, check phrases to cross threshold
    # ── TIER 2: Panic phrases ─────────────────────────────
    p_score, p_reasons = _phrase_score(text, words)
    score   += p_score
    reasons += p_reasons

    # ── TIER 3: Support words ─────────────────────────────
    support_hits = (words & SUPPORT_WORDS) - STOPWORDS
    s_pts = 2 * len(support_hits)
    score += s_pts
    if support_hits:
        reasons.append(f"support:{support_hits}")

    # ── NLP emotion ───────────────────────────────────────
    nlp_pts = 0
    if nlp_loaded and len(text.split()) >= 3:
        try:
            results = emotion_classifier(text)
            for em in results[0]:
                if em["label"] in ("fear", "anger") and em["score"] > 0.60:
                    nlp_pts += 5
                    reasons.append(f"NLP:{em['label']}({em['score']:.2f})")
                elif em["label"] == "sadness" and em["score"] > 0.70:
                    nlp_pts += 3
                    reasons.append(f"NLP:sadness({em['score']:.2f})")
        except Exception as e:
            print(f"NLP error: {e}")
    score += nlp_pts

    display_score = min(score, 10)
    print(f"SCORE={score}/{ALERT_THRESHOLD} | {reasons or 'no match'}")
    root.after(0, lambda s=display_score: _update_score_bar(s))

    if score >= ALERT_THRESHOLD:
        root.after(0, lambda: trigger_panic(text, score=score, reasons=reasons))
    else:
        root.after(3000, lambda: _update_score_bar(0))

# =========================================================
# EMERGENCY ALERT SENDER
# =========================================================

def send_emergency_sms(trigger_text):
    if not TWILIO_OK:
        root.after(0, lambda: messagebox.showerror(
            "Twilio Error", "Twilio is not configured correctly."))
        return False
    try:
        loc  = geocoder.ip("me")
        lat, lon = (loc.latlng if loc.latlng else (0.0, 0.0))
        link = f"https://maps.google.com/?q={lat},{lon}"
        msg  = (
            f"🚨 EMERGENCY ALERT 🚨\n\n"
            f"AI Panic Detector triggered.\n"
            f"Trigger: \"{trigger_text[:80]}\"\n\n"
            f"📍 Approx location: {link}\n\n"
            f"Please respond immediately."
        )
        if not VERIFIED_CONTACTS:
            root.after(0, lambda: messagebox.showwarning(
                "No Contacts",
                "No verified numbers in your Twilio account.\n"
                "Add them at twilio.com/console → Verified Caller IDs"))
            return False

        sent = 0
        for c in VERIFIED_CONTACTS:
            try:
                sms = twilio_client.messages.create(
                    body=msg, from_=TWILIO_NUMBER, to=c["phone"])
                print(f"SMS → {c['name']} ({c['phone']}) | SID: {sms.sid}")
                sent += 1
            except Exception as e:
                print(f"SMS failed to {c['phone']}: {e}")

        if sent > 0:
            root.after(0, lambda: (
                alert_count.__setitem__(0, alert_count[0] + 1),
                lbl_alerts.config(text=str(alert_count[0]))
            ))
            return True
        return False
    except Exception as e:
        err = str(e)
        root.after(0, lambda: messagebox.showerror("Alert Failed", err))
        return False

# =========================================================
# PANIC POPUP
# =========================================================

def show_panic_popup(text):
    pop = tk.Toplevel(root)
    pop.title("⚠ EMERGENCY DETECTED")
    pop.geometry("460x300")
    pop.configure(bg=SURFACE)
    pop.grab_set()
    pop.attributes("-topmost", True)

    tk.Frame(pop, bg=RED, height=4).pack(fill="x")
    tk.Label(pop, text="🚨", font=("Segoe UI Emoji", 48),
             fg=RED, bg=SURFACE).pack(pady=(16, 4))
    tk.Label(pop, text="PANIC DETECTED",
             font=("Courier New", 18, "bold"), fg=RED, bg=SURFACE).pack()
    tk.Label(pop, text=f'Detected: "{text[:80]}"',
             font=("Courier New", 10), fg=WHITE, bg=SURFACE,
             wraplength=380, justify="center").pack(pady=12)
    tk.Label(pop, text="Emergency alert SMS sent to all contacts.",
             font=("Courier New", 9), fg=GREEN_DIM, bg=SURFACE).pack()

    def _dismiss():
        pop.destroy()
        root.after(4000, _set_monitoring)

    btn = tk.Frame(pop, bg=RED, cursor="hand2")
    btn.pack(pady=14)
    lbl = tk.Label(btn, text="  ✓  DISMISS  ",
                   font=("Courier New", 12, "bold"),
                   fg=BG, bg=RED, padx=16, pady=10)
    lbl.pack()
    for w in (btn, lbl):
        w.bind("<Button-1>", lambda e: _dismiss())

    tk.Frame(pop, bg=RED, height=4).pack(fill="x", side="bottom")

# =========================================================
# TRIGGER PANIC — UNIFIED ENTRY POINT
# =========================================================

def trigger_panic(text, score=10, reasons=None):
    global last_alert_time
    now = time.time()
    if now - last_alert_time < ALERT_COOLDOWN:
        remaining = int(ALERT_COOLDOWN - (now - last_alert_time))
        print(f"Cooldown active — {remaining}s remaining")
        return
    last_alert_time = now

    _set_alert()
    alert_detail.config(text=f"⚠ Detected: \"{text[:60]}\"", fg=RED)
    _log(f"PANIC: {text}", tag="panic", score=score)

    cursor.execute(
        "INSERT INTO alert_log(timestamp, trigger, score, sent) VALUES(?,?,?,?)",
        (time.strftime("%Y-%m-%d %H:%M:%S"), text[:200], score, 1)
    )
    conn.commit()

    threading.Thread(target=_send_flow, args=(text,), daemon=True).start()

def _send_flow(text):
    ok = send_emergency_sms(text)
    if ok:
        root.after(0, lambda: show_panic_popup(text))
    else:
        root.after(0, lambda: _set_monitoring())

# =========================================================
# AUDIO PROCESSING
# =========================================================

def audio_callback(indata, _frames, _time, status):
    global current_volume
    data           = np.frombuffer(indata, dtype=np.int16)
    vol            = np.linalg.norm(data) * 0.02
    current_volume = vol
    if vol < 5:
        return
    audio_queue.put(bytes(indata))
    root.after(0, update_visualizer, vol)

_partial_alerted = set()

def listen_loop():
    """
    Main speech recognition loop.
    FIX: Partial results now also check PANIC_PHRASES (not just CRITICAL_WORDS),
         so "help me" heard partially triggers faster.
    """
    while running:
        try:
            data = audio_queue.get()

            if recognizer.AcceptWaveform(data):
                # ── Full result ───────────────────────────
                result = json.loads(recognizer.Result())
                text   = result.get("text", "").strip()
                _partial_alerted.clear()

                if len(text) >= 2:
                    normalized = _normalize(text)
                    root.after(0, lambda t=normalized: (
                        _log(t, tag="normal"),
                        alert_detail.config(text=f"🎤 {t}", fg=CYAN_DIM)
                    ))
                    threading.Thread(
                        target=analyze_text, args=(normalized,), daemon=True
                    ).start()

            else:
                # ── Partial result ────────────────────────
                partial = json.loads(recognizer.PartialResult())
                ptext   = partial.get("partial", "").strip().lower()
                if not ptext:
                    continue

                spoken   = set(ptext.split())

                # Check critical words in partial
                new_critical = (spoken & CRITICAL_WORDS) - _partial_alerted
                if new_critical:
                    _partial_alerted.update(new_critical)
                    word = next(iter(new_critical))
                    print(f"⚡ PARTIAL critical word: '{word}'")
                    repeat_now = _record_word_repetition(word)
                    root.after(0, lambda w=word, r=repeat_now: (
                        alert_detail.config(
                            text=f"⚡ Heard: '{w}'" + (" × AGAIN — alerting!" if r else " — say again to alert"),
                            fg=RED if r else AMBER),
                        _update_score_bar(10 if r else 5)
                    ))
                    if repeat_now:
                        root.after(0, lambda w=word: trigger_panic(w, score=10))
                    continue

                # FIX: Also check panic phrases in partial for faster response
                for phrase in PANIC_PHRASES:
                    if phrase in ptext and phrase not in _partial_alerted:
                        _partial_alerted.add(phrase)
                        print(f"⚡ PARTIAL panic phrase: '{phrase}'")
                        root.after(0, lambda p=phrase: (
                            alert_detail.config(text=f"⚡ Heard phrase: '{p}'", fg=RED),
                            _update_score_bar(8)
                        ))
                        root.after(0, lambda p=phrase: trigger_panic(p, score=8))
                        break

        except Exception as e:
            print(f"listen_loop error: {e}")

# =========================================================
# MANAGE CONTACTS WINDOW
# =========================================================

def _open_manage_contacts():
    w = tk.Toplevel(root)
    w.title("Manage Contacts")
    w.geometry("500x520")
    w.configure(bg=SURFACE)
    w.grab_set()

    tk.Frame(w, bg=CYAN, height=3).pack(fill="x")
    tk.Label(w, text="MANAGE CONTACTS",
             font=("Courier New", 14, "bold"),
             fg=WHITE, bg=SURFACE).pack(pady=(16, 2))
    tk.Label(w, text="Local emergency contacts database",
             font=("Courier New", 8), fg=MUTED, bg=SURFACE).pack(pady=(0, 10))

    list_outer = tk.Frame(w, bg=SURFACE2,
                          highlightbackground=BORDER, highlightthickness=1)
    list_outer.pack(fill="both", expand=True, padx=20, pady=(0, 8))

    hdr = tk.Frame(list_outer, bg=GHOST)
    hdr.pack(fill="x")
    for txt, wid, side in [("NAME", 18, "left"), ("PHONE", 16, "left"), ("ACTION", 8, "right")]:
        tk.Label(hdr, text=txt, font=("Courier New", 8, "bold"),
                 fg=MUTED, bg=GHOST, width=wid, anchor="w"
                 ).pack(side=side, padx=10, pady=6)

    list_canvas = tk.Canvas(list_outer, bg=SURFACE2, highlightthickness=0)
    scrollbar   = tk.Scrollbar(list_outer, orient="vertical", command=list_canvas.yview)
    list_inner  = tk.Frame(list_canvas, bg=SURFACE2)
    list_inner.bind("<Configure>", lambda e: list_canvas.configure(
        scrollregion=list_canvas.bbox("all")))
    list_canvas.create_window((0, 0), window=list_inner, anchor="nw")
    list_canvas.configure(yscrollcommand=scrollbar.set)
    list_canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def _rebuild():
        for child in list_inner.winfo_children():
            child.destroy()
        cursor.execute("SELECT id, name, phone FROM contacts ORDER BY id DESC")
        rows = cursor.fetchall()
        if not rows:
            tk.Label(list_inner, text="No contacts yet. Add one below.",
                     font=("Courier New", 9), fg=MUTED, bg=SURFACE2).pack(pady=16)
            return
        for cid, name, phone in rows:
            rf = tk.Frame(list_inner, bg=GHOST,
                          highlightbackground=BORDER, highlightthickness=1)
            rf.pack(fill="x", padx=6, pady=2)
            tk.Label(rf, text=name, font=("Courier New", 10, "bold"),
                     fg=WHITE, bg=GHOST, width=16, anchor="w").pack(side="left", padx=10, pady=6)
            tk.Label(rf, text=phone, font=("Courier New", 10),
                     fg=CYAN_DIM, bg=GHOST, width=14, anchor="w").pack(side="left")

            def _del(cid=cid):
                if messagebox.askyesno("Delete", "Remove this contact?", parent=w):
                    cursor.execute("DELETE FROM contacts WHERE id=?", (cid,))
                    conn.commit()
                    _rebuild()

            dl = tk.Label(rf, text=" ✕ DELETE ",
                          font=("Courier New", 9, "bold"),
                          fg=RED, bg=GHOST, cursor="hand2")
            dl.pack(side="right", padx=8)
            dl.bind("<Button-1>", lambda e, f=_del: f())

    _rebuild()

    tk.Frame(w, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(4, 0))
    add_sec = tk.Frame(w, bg=SURFACE)
    add_sec.pack(fill="x", padx=20, pady=8)
    tk.Label(add_sec, text="ADD NEW CONTACT",
             font=("Courier New", 8, "bold"), fg=CYAN, bg=SURFACE).pack(anchor="w", pady=(4, 6))

    fields = tk.Frame(add_sec, bg=SURFACE)
    fields.pack(fill="x")

    def _entry(parent, placeholder, width=15):
        e = tk.Entry(parent, bg=GHOST, fg=WHITE,
                     font=("Courier New", 10), relief="flat",
                     highlightthickness=1, highlightbackground=BORDER_LIT,
                     highlightcolor=CYAN, insertbackground=CYAN, width=width)
        e.pack(side="left", ipady=7, padx=(0, 6))
        e.insert(0, placeholder)
        e.bind("<FocusIn>",  lambda ev, p=placeholder: e.delete(0, "end") if e.get() == p else None)
        e.bind("<FocusOut>", lambda ev, p=placeholder: e.insert(0, p) if not e.get() else None)
        return e

    n_ent = _entry(fields, "Full Name")
    p_ent = _entry(fields, "+91… / +1…")

    def _save():
        nm = n_ent.get().strip()
        ph = p_ent.get().strip()
        if not nm or nm == "Full Name":
            messagebox.showwarning("Missing", "Enter a name.", parent=w); return
        if not ph or ph == "+91… / +1…":
            messagebox.showwarning("Missing", "Enter a phone number.", parent=w); return
        cursor.execute("INSERT INTO contacts(name, phone) VALUES(?,?)", (nm, ph))
        conn.commit()
        n_ent.delete(0, "end"); n_ent.insert(0, "Full Name")
        p_ent.delete(0, "end"); p_ent.insert(0, "+91… / +1…")
        _rebuild()

    ab = tk.Frame(fields, bg=CYAN, cursor="hand2")
    ab.pack(side="left")
    al = tk.Label(ab, text="  + ADD  ",
                  font=("Courier New", 10, "bold"),
                  fg=BG, bg=CYAN, padx=4, pady=7)
    al.pack()
    for wb in (ab, al):
        wb.bind("<Button-1>", lambda e: _save())

    tk.Frame(w, bg=CYAN, height=3).pack(fill="x", side="bottom")

# =========================================================
# ALERT HISTORY WINDOW
# =========================================================

def _open_alert_history():
    w = tk.Toplevel(root)
    w.title("Alert History")
    w.geometry("640x420")
    w.configure(bg=SURFACE)
    w.grab_set()

    tk.Frame(w, bg=RED, height=3).pack(fill="x")
    tk.Label(w, text="ALERT HISTORY",
             font=("Courier New", 14, "bold"),
             fg=WHITE, bg=SURFACE).pack(pady=(14, 2))

    txt = tk.Text(w, bg=SURFACE2, fg=WHITE,
                  font=("Courier New", 9), relief="flat",
                  highlightthickness=0, wrap="word",
                  spacing1=3, spacing3=3)
    txt.pack(fill="both", expand=True, padx=20, pady=10)
    txt.tag_config("ts",  foreground=MUTED)
    txt.tag_config("trg", foreground=RED)
    txt.tag_config("sc",  foreground=AMBER)

    cursor.execute(
        "SELECT timestamp, trigger, score, sent FROM alert_log ORDER BY id DESC LIMIT 100")
    rows = cursor.fetchall()
    if not rows:
        txt.insert("end", "No alerts recorded yet.\n", "ts")
    else:
        for ts, trg, score, sent in rows:
            txt.insert("end", f"[{ts}]  ", "ts")
            txt.insert("end", f"score={score}  ", "sc")
            txt.insert("end", f"sent={'YES' if sent else 'NO'}  ", "ts")
            txt.insert("end", f"{trg}\n", "trg")

    txt.config(state="disabled")
    tk.Frame(w, bg=RED, height=3).pack(fill="x", side="bottom")

# =========================================================
# START SYSTEM
# =========================================================

def start_system():
    global audio_stream
    try:
        audio_stream = sd.RawInputStream(
            samplerate=16000, blocksize=2000,
            dtype="int16", channels=1,
            latency="low", callback=audio_callback
        )
        audio_stream.start()
        statusbar_lbl.config(text="● SYSTEM ONLINE  |  MICROPHONE ACTIVE  |  LISTENING")
        _log("System started. Listening for panic speech…", tag="info")
        threading.Thread(target=listen_loop, daemon=True).start()
        threading.Thread(target=fetch_verified_numbers, daemon=True).start()
    except Exception as e:
        messagebox.showerror("Microphone Error", str(e))
        statusbar_lbl.config(text="⚠ MICROPHONE ERROR", fg=RED)
        lbl_mic.config(text="● ERROR", fg=RED)

def _update_badges():
    if DATASET_LOADED:
        dataset_badge.config(
            text=f"  CSV: {DATASET_PHRASES} PHRASES  ",
            fg=GREEN, bg="#0a1f12")
    else:
        dataset_badge.config(
            text="  CSV: NOT FOUND  ",
            fg=AMBER, bg="#1f1505")
    lbl_phrases.config(text=str(len(PANIC_PHRASES)))

def _on_close():
    global running
    running = False
    if audio_stream:
        try:
            audio_stream.stop()
            audio_stream.close()
        except Exception:
            pass
    conn.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", _on_close)

root.after(400,  start_system)
root.after(600,  _update_badges)
threading.Thread(target=load_nlp_model, daemon=True).start()

root.mainloop()