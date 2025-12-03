import cv2
import mediapipe as mp
import pygame
import numpy as np
import random
import math

# --- Constants ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
BACKGROUND_COLOR = (15, 20, 40)

# --- UI and Colors ---
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
KEY_PRESS_COLORS = [
    (255, 87, 87), (255, 167, 87), (255, 250, 87),
    (87, 255, 127), (87, 187, 255)
]

# --- Gameplay ---
DEBOUNCE_FRAMES = 3

# --- Particle System ---
PARTICLES = []

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-6, -1)
        self.lifespan = random.randint(18, 40)
        self.radius = random.randint(3, 7)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.15  # gravity
        self.lifespan -= 1
        self.radius -= 0.08
        if self.radius < 0: self.radius = 0

    def draw(self, screen):
        alpha = int(255 * (self.lifespan / 40))
        if alpha > 0 and self.radius > 0:
            surface = pygame.Surface((int(self.radius * 2), int(self.radius * 2)), pygame.SRCALPHA)
            pygame.draw.circle(surface, self.color + (alpha,), (int(self.radius), int(self.radius)), int(self.radius))
            screen.blit(surface, (self.x - self.radius, self.y - self.radius))

# --- Drum Sound Generation (procedural) ---
def make_stereo(sound_array):
    """Convert mono numpy array int16 to stereo shape (n,2)."""
    if sound_array.ndim == 1:
        return np.ascontiguousarray(np.vstack((sound_array, sound_array)).T)
    return sound_array

def generate_kick(duration=0.5, sample_rate=44100, amplitude=0.9):
    # Kick: sine sweep down with fast decay
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    start_freq = 150.0
    end_freq = 50.0
    freqs = np.linspace(start_freq, end_freq, n)
    wave = np.sin(2 * np.pi * freqs * t)
    env = np.exp(-8 * t)  # fast decay
    wave *= env
    wave = wave * amplitude
    wave = (wave / np.max(np.abs(wave)) * (2**15 - 1)).astype(np.int16)
    return pygame.sndarray.make_sound(make_stereo(wave))

def generate_snare(duration=0.35, sample_rate=44100, amplitude=0.8):
    # Snare: noise burst + short metallic sine
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    noise = np.random.normal(0, 1, n)
    env_noise = np.exp(-15 * t) * (1 - np.exp(-50 * t))  # quick attack, medium decay
    noise *= env_noise
    # Add short sine body
    sine = 0.6 * np.sin(2 * np.pi * 200 * t) * np.exp(-6 * t)
    wave = (noise + sine)
    wave *= amplitude
    wave = (wave / np.max(np.abs(wave)) * (2**15 - 1)).astype(np.int16)
    return pygame.sndarray.make_sound(make_stereo(wave))

def generate_hihat(duration=0.12, sample_rate=44100, amplitude=0.6):
    # Hi-hat / closed hat: very short bright noise
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    noise = np.random.normal(0, 1, n)
    env = np.exp(-40 * t)  # very fast decay
    # emphasize high frequencies by differentiating noise
    noise = np.concatenate([[0], np.diff(noise)]) * env
    wave = noise * amplitude
    wave = (wave / (np.max(np.abs(wave)) + 1e-9) * (2**15 - 1)).astype(np.int16)
    return pygame.sndarray.make_sound(make_stereo(wave))

def generate_tom(duration=0.45, sample_rate=44100, amplitude=0.85, freq=120):
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    wave = np.sin(2 * np.pi * freq * t) * np.exp(-4 * t)  # medium decay
    # add some harmonic
    wave += 0.3 * np.sin(2 * np.pi * (freq * 2) * t) * np.exp(-5 * t)
    wave *= amplitude
    wave = (wave / np.max(np.abs(wave)) * (2**15 - 1)).astype(np.int16)
    return pygame.sndarray.make_sound(make_stereo(wave))

def generate_crash(duration=1.0, sample_rate=44100, amplitude=0.8):
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    noise = np.random.normal(0, 1, n)
    env = np.exp(-3 * t)  # longer decay for crash
    # add metallic resonances
    resonances = np.zeros_like(noise)
    for f in [3000, 4500, 6000]:
        resonances += 0.2 * np.sin(2 * np.pi * f * t) * np.exp(-6 * t)
    wave = (noise * env * 0.8) + resonances
    wave *= amplitude
    wave = (wave / np.max(np.abs(wave)) * (2**15 - 1)).astype(np.int16)
    return pygame.sndarray.make_sound(make_stereo(wave))

# --- Finger Counting Logic (same as your piano) ---
def count_fingers(hand_landmarks, hand_label):
    if not hand_landmarks: return 0
    tip_ids = [4, 8, 12, 16, 20]
    fingers_up = 0
    # Thumb logic (x comparison depends on left/right)
    if hand_label == 'Right':
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 2].x:
            fingers_up += 1
    else:
        if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 2].x:
            fingers_up += 1
    # Fingers
    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            fingers_up += 1
    return fingers_up

# --- Background & UI Drawing ---
STARS = [(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT), random.randint(1, 2)) for _ in range(150)]
def draw_background(screen):
    screen.fill(BACKGROUND_COLOR)
    for x, y, r in STARS:
        brightness = random.randint(100, 255)
        pygame.draw.circle(screen, (brightness, brightness, brightness), (x, y), r)

def draw_circular_webcam(screen, frame):
    frame_height, frame_width, _ = frame.shape
    cam_diameter = 300
    frame_small = cv2.resize(frame, (cam_diameter, cam_diameter))
    frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    cam_surface = pygame.surfarray.make_surface(np.rot90(frame_small))
    mask_surface = pygame.Surface((cam_diameter, cam_diameter), pygame.SRCALPHA)
    pygame.draw.circle(mask_surface, (255, 255, 255, 255), (cam_diameter // 2, cam_diameter // 2), cam_diameter // 2)
    cam_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    pos_x = SCREEN_WIDTH // 2 - cam_diameter // 2
    pos_y = 150
    pygame.draw.circle(screen, KEY_PRESS_COLORS[4], (pos_x + cam_diameter//2, pos_y + cam_diameter//2), cam_diameter//2 + 8)
    screen.blit(cam_surface, (pos_x, pos_y))

def draw_drum_pad_visuals(screen, pad_names, active_pads):
    # draw circular pads across the bottom
    pad_count = len(pad_names)
    spacing = (SCREEN_WIDTH - 200) / pad_count
    radius = 60
    start_x = 100 + spacing / 2
    y = SCREEN_HEIGHT - 160
    font = pygame.font.Font(None, 28)
    for i, name in enumerate(pad_names):
        x = int(start_x + i * spacing)
        is_active = name in active_pads
        color = KEY_PRESS_COLORS[i % len(KEY_PRESS_COLORS)] if is_active else (230, 230, 235)
        pygame.draw.circle(screen, (40, 40, 40), (x, y+8), radius+10)  # shadow
        pygame.draw.circle(screen, color, (x, y), radius)
        label = font.render(name, True, BLACK_COLOR)
        screen.blit(label, (x - label.get_width()//2, y - 10))

# --- Main Application ---
def run_drum_app():
    pygame.init()
    # Use 2 channels stereo; buffer small for low latency
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Magical Hand Drum")
    title_font = pygame.font.Font(None, 72)
    info_font = pygame.font.Font(None, 36)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Generate drum sounds
    drum_sounds = {
        'Kick': generate_kick(),
        'Snare': generate_snare(),
        'HiHat': generate_hihat(),
        'Tom': generate_tom(freq=140),
        'Crash': generate_crash()
    }
    pad_order = ['Kick', 'Snare', 'HiHat', 'Tom', 'Crash']

    # Mapping: Left hand fingers 1-5 -> pads 0-4; Right hand offset (same mapping for now)
    # State for debounce
    stable_fingers = {'Left': 0, 'Right': 0}
    pending_fingers = {'Left': 0, 'Right': 0}
    debounce_counters = {'Left': 0, 'Right': 0}

    global PARTICLES

    running = True
    clock = pygame.time.Clock()
    while running:
        # input events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        raw_fingers = {'Left': 0, 'Right': 0}
        active_pads = set()

        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = handedness.classification[0].label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                raw_fingers[hand_label] = count_fingers(hand_landmarks, hand_label)

        # debounce + trigger logic
        for hand_label in ['Left', 'Right']:
            if raw_fingers[hand_label] == pending_fingers[hand_label]:
                debounce_counters[hand_label] += 1
            else:
                pending_fingers[hand_label] = raw_fingers[hand_label]
                debounce_counters[hand_label] = 0

            if debounce_counters[hand_label] >= DEBOUNCE_FRAMES:
                if stable_fingers[hand_label] != pending_fingers[hand_label]:
                    stable_fingers[hand_label] = pending_fingers[hand_label]
                    # trigger when fingers > 0
                    if stable_fingers[hand_label] > 0:
                        offset = 0 if hand_label == 'Left' else 0  # both hands map to same 5 pads here
                        pad_idx = offset + stable_fingers[hand_label] - 1
                        if 0 <= pad_idx < len(pad_order):
                            pad_name = pad_order[pad_idx]
                            # play sound (use mixer channels to allow overlap)
                            try:
                                drum_sounds[pad_name].play()
                            except Exception:
                                pass
                            # spawn particles around pad position
                            pad_x = 100 + (SCREEN_WIDTH - 200) / len(pad_order) * pad_idx + ((SCREEN_WIDTH - 200) / len(pad_order)) / 2
                            for _ in range(25):
                                PARTICLES.append(Particle(pad_x, SCREEN_HEIGHT - 160, KEY_PRESS_COLORS[pad_idx % len(KEY_PRESS_COLORS)]))

            # Visual active indicator
            if stable_fingers[hand_label] > 0:
                pad_idx = stable_fingers[hand_label] - 1
                if 0 <= pad_idx < len(pad_order):
                    active_pads.add(pad_order[pad_idx])

        # --- Drawing ---
        draw_background(screen)
        draw_drum_pad_visuals(screen, pad_order, active_pads)

        # particles
        for p in PARTICLES:
            p.update()
        PARTICLES = [p for p in PARTICLES if p.lifespan > 0]
        for p in PARTICLES:
            p.draw(screen)

        draw_circular_webcam(screen, frame)

        # UI Text
        title_text = title_font.render("Magical Hand Drum", True, WHITE_COLOR)
        info_l = info_font.render(f"Left Fingers: {stable_fingers['Left']}", True, WHITE_COLOR)
        info_r = info_font.render(f"Right Fingers: {stable_fingers['Right']}", True, WHITE_COLOR)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() / 2, 40))
        screen.blit(info_l, (100, 60))
        screen.blit(info_r, (SCREEN_WIDTH - 100 - info_r.get_width(), 60))

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()
    print("Application closed successfully.")

if __name__ == '__main__':
    run_drum_app()
