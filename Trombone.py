import cv2
import mediapipe as mp
import pygame
import numpy as np
import random
import math
import time

# --- Constants ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
BACKGROUND_COLOR = (15, 20, 40)  # deep night blue

# --- UI Colors & Fonts ---
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
KEY_PRESS_COLORS = [
    (255, 87, 87), (255, 167, 87), (255, 250, 87),
    (87, 255, 127), (87, 187, 255)
]

# --- Gameplay ---
DEBOUNCE_FRAMES = 3  # number of frames a gesture must be stable to register

# --- Particles ---
PARTICLES = []

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-2.5, 2.5)
        self.vy = random.uniform(-5, -1)
        self.lifespan = random.randint(20, 42)
        self.radius = random.randint(3, 7)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.12  # gravity
        self.lifespan -= 1
        self.radius -= 0.08
        if self.radius < 0:
            self.radius = 0

    def draw(self, screen):
        alpha = int(255 * (self.lifespan / 42))
        if alpha > 0 and self.radius > 0:
            surface = pygame.Surface((int(self.radius * 2), int(self.radius * 2)), pygame.SRCALPHA)
            pygame.draw.circle(surface, self.color + (alpha,), (int(self.radius), int(self.radius)), int(self.radius))
            screen.blit(surface, (self.x - self.radius, self.y - self.radius))

# --- Utilities for sound generation ---
def make_stereo(sound_mono):
    if sound_mono.ndim == 1:
        return np.ascontiguousarray(np.vstack((sound_mono, sound_mono)).T)
    return sound_mono

# --- Trombone-like sound generator ---
def generate_trombone_wave(base_freq, duration=1.0, sample_rate=44100, amplitude=0.6, slide_cent=0):
    """
    Generate a trombone-like tone:
      - base_freq: fundamental frequency in Hz
      - duration: seconds
      - slide_cent: number of cents to shift (positive = up, negative = down) applied across duration (adds portamento)
    Returns a pygame Sound object.
    """
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    # Create a smooth pitch slide using cents
    cents = np.linspace(0, slide_cent, n)
    freq_multiplier = 2 ** (cents / 1200.0)
    instantaneous_freq = base_freq * freq_multiplier

    # integrate instantaneous frequency to phase
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate

    # rich harmonic content: fundamental + several odd/even partials with decays
    wave = 0.0
    harmonics = [1.0, 0.56, 0.28, 0.14, 0.08]  # trombone-ish harmonic balance
    for h_amp, h_mult in zip(harmonics, [1, 2, 3, 4, 5]):
        wave += h_amp * np.sin(h_mult * phase)

    # Apply a slow attack + medium decay envelope to simulate bowing / lip buzz
    attack_time = max(0.02, 0.08 * (1 - amplitude))  # shorter attack for louder notes
    env = np.ones_like(t)
    attack_samples = int(sample_rate * attack_time)
    if attack_samples > 0:
        env[:attack_samples] = np.linspace(0, 1.0, attack_samples)
    env *= np.exp(-2.2 * t)  # overall gentle decay so note doesn't ring forever

    # Add subtle vibrato (low-rate pitch modulation) for realism
    vibrato = 1 + 0.0025 * np.sin(2 * np.pi * 5.2 * t)  # tiny amplitude, ~5 Hz
    wave *= env * vibrato

    # Gentle lowpass-ish smoothing by convolving a small kernel (to tame harsh harmonics)
    kernel = np.array([0.25, 0.5, 0.25])
    wave = np.convolve(wave, kernel, mode='same')

    # Normalize and scale
    if np.max(np.abs(wave)) > 0:
        wave = wave / np.max(np.abs(wave))
    wave = (wave * amplitude * (2**15 - 1)).astype(np.int16)

    stereo = make_stereo(wave)
    return pygame.sndarray.make_sound(stereo)

# --- Finger counting (same logic you used) ---
def count_fingers(hand_landmarks, hand_label):
    if not hand_landmarks:
        return 0
    tip_ids = [4, 8, 12, 16, 20]
    fingers_up = 0
    # thumb: x comparison depends on handedness
    try:
        if hand_label == 'Right':
            if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 2].x:
                fingers_up += 1
        else:
            if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 2].x:
                fingers_up += 1
    except:
        pass
    # other fingers: compare y of tip vs pip-joint
    for i in range(1, 5):
        try:
            if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
                fingers_up += 1
        except:
            pass
    return fingers_up

# --- UI drawing functions (background, webcam, pads) ---
STARS = [(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT), random.randint(1, 2)) for _ in range(150)]

def draw_background(screen):
    screen.fill(BACKGROUND_COLOR)
    for x, y, r in STARS:
        brightness = random.randint(110, 255)
        pygame.draw.circle(screen, (brightness, brightness, brightness), (x, y), r)

def draw_circular_webcam(screen, frame):
    cam_diameter = 300
    frame_small = cv2.resize(frame, (cam_diameter, cam_diameter))
    frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    cam_surface = pygame.surfarray.make_surface(np.rot90(frame_small))
    mask_surface = pygame.Surface((cam_diameter, cam_diameter), pygame.SRCALPHA)
    pygame.draw.circle(mask_surface, (255, 255, 255, 255), (cam_diameter // 2, cam_diameter // 2), cam_diameter // 2)
    cam_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    pos_x = SCREEN_WIDTH // 2 - cam_diameter // 2
    pos_y = 150
    pygame.draw.circle(screen, KEY_PRESS_COLORS[4], (pos_x + cam_diameter // 2, pos_y + cam_diameter // 2), cam_diameter // 2 + 8)
    screen.blit(cam_surface, (pos_x, pos_y))

def draw_trombone_ui(screen, note_names, active_note):
    # Draw horizontal note strip near bottom to show selected note
    strip_w = SCREEN_WIDTH - 200
    start_x = 100
    y = SCREEN_HEIGHT - 180
    key_w = strip_w / len(note_names)
    font = pygame.font.Font(None, 30)
    for i, n in enumerate(note_names):
        rect = pygame.Rect(int(start_x + i * key_w), y, int(key_w - 8), 110)
        color = KEY_PRESS_COLORS[i % len(KEY_PRESS_COLORS)] if n == active_note else (240, 240, 245)
        pygame.draw.rect(screen, (35, 35, 35), rect.move(0, 6), border_radius=12)  # shadow
        pygame.draw.rect(screen, color, rect, border_radius=12)
        label = font.render(n, True, BLACK_COLOR)
        screen.blit(label, (rect.centerx - label.get_width() // 2, rect.bottom - 40))

# --- Main Application ---
def run_trombone_app():
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Magical Hand Trombone")
    title_font = pygame.font.Font(None, 72)
    info_font = pygame.font.Font(None, 32)

    # Mediapipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Trombone note mapping (choose 5 comfortable trombone positions/fundamentals)
    # You can change these frequencies to taste (Hz)
    note_names = ["Bb2", "C3", "D3", "F3", "Bb3"]  # descriptive labels
    note_freqs = [116.54, 130.81, 146.83, 174.61, 233.08]  # base frequencies (example)

    # Pre-generate short base sounds (we will apply slide by regenerating small slide segments when playing)
    # But to avoid heavy CPU regen each frame, we will generate full note bodies with zero slide and use
    # a separate short slide blip to simulate quick pitch slides when triggered.
    base_sounds = [generate_trombone_wave(freq, duration=1.2, amplitude=0.65, slide_cent=0) for freq in note_freqs]

    # state for debounce
    stable_fingers = {'Left': 0, 'Right': 0}
    pending_fingers = {'Left': 0, 'Right': 0}
    debounce_counters = {'Left': 0, 'Right': 0}

    # For portamento: remember currently playing channel and target frequency
    current_play = {'channel': None, 'start_time': 0, 'freq': None}

    global PARTICLES

    clock = pygame.time.Clock()
    running = True
    while running:
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
        active_note = None

        # We'll also compute slide control: distance between thumb tip (4) and index tip (8) of the active hand
        slide_control = None

        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = handedness.classification[0].label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                raw_fingers[hand_label] = count_fingers(hand_landmarks, hand_label)
                # compute thumb-index pixel distance for slide (if available)
                try:
                    thumb = hand_landmarks.landmark[4]
                    index = hand_landmarks.landmark[8]
                    # use x-distance normalized to frame width
                    frame_h, frame_w, _ = frame.shape
                    dx = (thumb.x - index.x) * frame_w
                    dy = (thumb.y - index.y) * frame_h
                    dist = math.hypot(dx, dy)
                    # store last seen slide_control (use the hand that triggered sound)
                    slide_control = dist
                except:
                    pass

        # Debounce and trigger logic per hand
        for hand_label in ['Left', 'Right']:
            if raw_fingers[hand_label] == pending_fingers[hand_label]:
                debounce_counters[hand_label] += 1
            else:
                pending_fingers[hand_label] = raw_fingers[hand_label]
                debounce_counters[hand_label] = 0

            if debounce_counters[hand_label] >= DEBOUNCE_FRAMES:
                if stable_fingers[hand_label] != pending_fingers[hand_label]:
                    stable_fingers[hand_label] = pending_fingers[hand_label]
                    # trigger only when fingers > 0
                    if stable_fingers[hand_label] > 0:
                        # mapping: 1..5 fingers -> note index 0..4
                        note_idx = stable_fingers[hand_label] - 1
                        if 0 <= note_idx < len(note_freqs):
                            target_freq = note_freqs[note_idx]
                            active_note = note_names[note_idx]
                            # compute slide amount in cents based on slide_control (if present)
                            # We map a reasonable pixel distance to +/- 150 cents (1.5 semitones)
                            slide_cent = 0
                            if slide_control is not None:
                                # normalize slide_control roughly between 20..220
                                sc = max(0.0, min(220.0, slide_control))
                                slide_cent = int((sc - 60.0) / 160.0 * 300.0)  # -~112 to +~235 cents depending on distance
                                # clamp
                                slide_cent = max(-300, min(300, slide_cent))
                            # create a short trombone note with an initial slide (simulate immediate portamento)
                            # For lower CPU, reuse base_sounds but play a short slide 'attack' followed by base sound.
                            # We'll generate a short slide blip and play it, then play the base sound
                            try:
                                # short slide blip: 0.18s blending from +/- slide_cent to 0
                                blip = generate_trombone_wave(target_freq, duration=0.18, amplitude=0.7, slide_cent=slide_cent)
                                blip.play()
                                # small delay before base sustain to let blip lead into it (non-blocking)
                                # play base sound on another channel allowing overlap
                                base_snd = base_sounds[note_idx]
                                base_snd.play(-1)  # loop to simulate sustained trombone until finger is released
                                current_play['channel'] = base_snd
                                current_play['start_time'] = time.time()
                                current_play['freq'] = target_freq
                            except Exception as e:
                                # ignore any pygame playback errors
                                # print("Sound play error:", e)
                                pass

                            # spawn particles under UI pad for visual feedback
                            pad_x = 100 + (SCREEN_WIDTH - 200) / len(note_names) * note_idx + ((SCREEN_WIDTH - 200) / len(note_names)) / 2
                            for _ in range(28):
                                PARTICLES.append(Particle(pad_x, SCREEN_HEIGHT - 120, KEY_PRESS_COLORS[note_idx % len(KEY_PRESS_COLORS)]))

        # When fingers are released (0), stop sustained base sound(s)
        # if both hands stable_fingers are 0 => stop any looped base sounds
        if stable_fingers['Left'] == 0 and stable_fingers['Right'] == 0:
            # Note: base_sounds were played with play(-1) above; stop all channels associated with them
            for snd in base_sounds:
                try:
                    snd.stop()
                except:
                    pass
            current_play['channel'] = None
            current_play['freq'] = None
            active_note = None
        else:
            # if one of the hands has a stable finger count, indicate that note visually
            for hand_label in ['Left', 'Right']:
                if stable_fingers[hand_label] > 0:
                    idx = stable_fingers[hand_label] - 1
                    if 0 <= idx < len(note_names):
                        active_note = note_names[idx]

        # --- Drawing ---
        draw_background(screen)
        draw_trombone_ui(screen, note_names, active_note)

        # update particles
        for p in PARTICLES:
            p.update()
        PARTICLES = [p for p in PARTICLES if p.lifespan > 0]
        for p in PARTICLES:
            p.draw(screen)

        draw_circular_webcam(screen, frame)

        # UI text
        title_text = title_font.render("Magical Hand Trombone", True, WHITE_COLOR)
        info_l = info_font.render(f"Left: {stable_fingers['Left']}", True, WHITE_COLOR)
        info_r = info_font.render(f"Right: {stable_fingers['Right']}", True, WHITE_COLOR)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 40))
        screen.blit(info_l, (100, 60))
        screen.blit(info_r, (SCREEN_WIDTH - 100 - info_r.get_width(), 60))

        pygame.display.flip()
        clock.tick(60)

    # cleanup
    cap.release()
    pygame.quit()
    print("Application closed successfully.")

if __name__ == '__main__':
    run_trombone_app()
