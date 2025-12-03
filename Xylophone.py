import cv2
import mediapipe as mp
import pygame
import numpy as np
import random
import math

# ---------------- Constants ----------------
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720

WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)

BAR_BASE_COLOR = (230, 230, 235)
BAR_SHADOW = (140, 140, 160)
HIT_COLORS = [
    (255, 100, 100), (255, 165, 80), (255, 230, 90), (120, 220, 160)
]

DEBOUNCE_FRAMES = 3
PARTICLES = []


# ---------------- PARTICLE EFFECT ----------------
class Particle:
    def __init__(self, x, y, color):
        self.x = x + random.uniform(-30, 30)
        self.y = y + random.uniform(-10, 10)
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-5, -1.5)
        self.color = color
        self.lifespan = random.randint(18, 36)
        self.radius = random.uniform(3, 7)

    def update(self):
        self.vy += 0.15
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.radius *= 0.97

    def draw(self, screen):
        alpha = int(255 * max(0, self.lifespan / 36))
        if alpha <= 0 or self.radius <= 0:
            return

        size = int(self.radius * 2) + 2
        surface = pygame.Surface((size, size), pygame.SRCALPHA)

        r, g, b = self.color
        pygame.draw.circle(surface, (r, g, b, alpha),
                           (size // 2, size // 2), max(1, int(self.radius)))

        screen.blit(surface, (self.x - self.radius, self.y - self.radius))


# ---------------- XYLOPHONE SOUND ----------------
def generate_xylophone_hit(freq, duration=0.9, sample_rate=44100, amplitude=0.6):
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False)

    decay_rate = 6.0
    env = np.exp(-decay_rate * t)
    attack = 1.0 - np.exp(-30 * t)
    envelope = attack * env

    wave = np.zeros_like(t)
    partials = [
        (1.0, 1.0),
        (0.56, 2.7),
        (0.34, 3.9),
        (0.18, 5.5),
        (0.09, 8.2)
    ]

    for amp, mult in partials:
        wave += amp * np.sin(2 * np.pi * freq * mult * t)

    noise = np.random.normal(0, 1, n)
    strike = noise * np.exp(-120 * t) * 0.5

    final = (wave + strike) * envelope
    final = final / np.max(np.abs(final))

    final = (final * (2**15 - 1) * amplitude).astype(np.int16)
    stereo = np.vstack((final, final)).T.copy(order="C")

    return pygame.sndarray.make_sound(stereo)


# ---------------- FINGER COUNT ----------------
def count_fingers(hand_landmarks, hand_label):
    tip_ids = [4, 8, 12, 16, 20]
    fingers_up = 0

    # Thumb
    if hand_label == "Right":
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x:
            fingers_up += 1
    else:
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[2].x:
            fingers_up += 1

    # Other fingers  
    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            fingers_up += 1

    return fingers_up


# ---------------- BACKGROUNDS ----------------
bg_shift = 0

def clamp_int(value):
    return max(0, min(255, int(value)))

def draw_background_aurora(screen):
    global bg_shift
    bg_shift += 0.008

    for y in range(SCREEN_HEIGHT):
        s1 = math.sin(0.02 * y + bg_shift)
        s2 = math.sin(0.025 * y + bg_shift * 1.5)
        s3 = math.sin(0.018 * y + bg_shift * 2)

        r = clamp_int(100 + 80 * s1)
        g = clamp_int(60 + 120 * s2)
        b = clamp_int(150 + 100 * s3)

        pygame.draw.line(screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))


def draw_background_simple(screen):
    for y in range(SCREEN_HEIGHT):
        t = y / SCREEN_HEIGHT
        r = int(30 + 140 * t)
        g = int(80 + 110 * (1 - t))
        b = int(160 + 90 * (1 - t))

        pygame.draw.line(screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))


# ---------------- WEBCAM CIRCLE ----------------
def draw_circular_webcam(screen, frame):
    cam_diameter = 260

    frame = cv2.resize(frame, (cam_diameter, cam_diameter))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cam_surface = pygame.surfarray.make_surface(np.rot90(frame))

    mask = pygame.Surface((cam_diameter, cam_diameter), pygame.SRCALPHA)
    pygame.draw.circle(mask, (255, 255, 255, 255),
                       (cam_diameter // 2, cam_diameter // 2),
                       cam_diameter // 2)

    cam_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    x = SCREEN_WIDTH - cam_diameter - 40
    y = 120

    pygame.draw.circle(screen, (255, 180, 90),
                       (x + cam_diameter // 2, y + cam_diameter // 2),
                       cam_diameter // 2 + 8)

    screen.blit(cam_surface, (x, y))


# ---------------- XYLOPHONE UI ----------------
def draw_xylophone(screen, bar_names, active_bars):
    bar_width = SCREEN_WIDTH - 280
    bar_height = 48
    spacing = 18
    base_x = 120
    base_y = 140

    for i, name in enumerate(bar_names):
        y = base_y + i * (bar_height + spacing)

        rect = pygame.Rect(base_x, y, bar_width, bar_height)
        shadow = rect.move(6, 6)

        is_active = i in active_bars

        pygame.draw.rect(screen, BAR_SHADOW, shadow, border_radius=10)

        color = HIT_COLORS[i % len(HIT_COLORS)] if is_active else BAR_BASE_COLOR

        pygame.draw.rect(screen, color, rect, border_radius=10)
        pygame.draw.rect(screen, BLACK_COLOR, rect, 2, border_radius=10)

        font = pygame.font.Font(None, 28)
        txt = font.render(name, True, BLACK_COLOR)
        screen.blit(txt,
                    (rect.right - 40 - txt.get_width() // 2,
                     rect.centery - txt.get_height() // 2))


# ---------------- MAIN APPLICATION ----------------
def run_xylophone_app():
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Magical Hand Xylophone")

    title_font = pygame.font.Font(None, 72)
    info_font = pygame.font.Font(None, 36)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    # Xylophone notes
    notes_freq = [261.63, 293.66, 329.63, 349.23,
                  392.00, 440.00, 493.88, 523.25]
    bar_names = ["C4", "D4", "E4", "F4",
                 "G4", "A4", "B4", "C5"]

    sounds = [generate_xylophone_hit(f) for f in notes_freq]

    stable_fingers = {'Left': 0, 'Right': 0}
    pending_fingers = {'Left': 0, 'Right': 0}
    counters = {'Left': 0, 'Right': 0}

    background_mode = 1  # remove UI text, keep switching capability

    global PARTICLES
    clock = pygame.time.Clock()

    running = True
    while running:

        # EVENTS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key in (pygame.K_1, pygame.K_2):
                    background_mode = int(event.unicode)

        # WEBCAM
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

        else:
            frame = None
            result = None

        raw_fingers = {'Left': 0, 'Right': 0}
        active_bars = set()

        if result and result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                    result.multi_hand_landmarks,
                    result.multi_handedness):

                label = handedness.classification[0].label
                raw_fingers[label] = count_fingers(hand_landmarks, label)

        # DEBOUNCE LOGIC
        for hand in ("Left", "Right"):

            if raw_fingers[hand] == pending_fingers[hand]:
                counters[hand] += 1
            else:
                pending_fingers[hand] = raw_fingers[hand]
                counters[hand] = 0

            if counters[hand] >= DEBOUNCE_FRAMES:
                if stable_fingers[hand] != pending_fingers[hand]:
                    stable_fingers[hand] = pending_fingers[hand]

                    if stable_fingers[hand] > 0:
                        offset = 0 if hand == "Left" else 4
                        idx = offset + stable_fingers[hand] - 1

                        if 0 <= idx < 8:
                            sounds[idx].play()

                            cx = 120 + (SCREEN_WIDTH - 280) // 2
                            cy = 140 + idx * (48 + 18) + 24

                            for _ in range(20):
                                PARTICLES.append(
                                    Particle(cx, cy,
                                             HIT_COLORS[idx % 4])
                                )

            if stable_fingers[hand] > 0:
                idx = (0 if hand == "Left" else 4) + stable_fingers[hand] - 1
                if 0 <= idx < 8:
                    active_bars.add(idx)

        # DRAW BACKGROUND
        if background_mode == 1:
            draw_background_aurora(screen)
        else:
            draw_background_simple(screen)

        # DRAW XYLOPHONE
        draw_xylophone(screen, bar_names, active_bars)

        # DRAW PARTICLES
        for p in PARTICLES:
            p.update()
        PARTICLES = [p for p in PARTICLES if p.lifespan > 0]
        for p in PARTICLES:
            p.draw(screen)

        # DRAW WEBCAM
        if frame is not None:
            draw_circular_webcam(screen, frame)

        # TITLE
        title = title_font.render("Magical Hand Xylophone", True, WHITE_COLOR)
        screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 30))

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pygame.quit()


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    run_xylophone_app()
