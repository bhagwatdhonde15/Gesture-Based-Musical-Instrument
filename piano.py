import cv2
import mediapipe as mp
import pygame
import numpy as np
import random  

# --- Constants ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
BACKGROUND_COLOR = (15, 20, 40) # A deep night blue

# --- UI and Fonts ---
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)

# --- Piano Colors ---
KEY_WHITE_COLOR = (245, 245, 255)
KEY_WHITE_SHADOW = (180, 180, 200)
KEY_PRESS_COLORS = [ # A vibrant, coordinated palette
    (255, 87, 87), (255, 167, 87), (255, 250, 87), 
    (87, 255, 127), (87, 187, 255)
]

# --- Gameplay ---
DEBOUNCE_FRAMES = 3 # Number of frames a gesture must be held to register

# --- Particle System ---
PARTICLES = [] # This is the global list of particles

class Particle:

    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-4, -1)
        self.lifespan = random.randint(20, 40)
        self.radius = random.randint(3, 7)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.radius -= 0.1
        if self.radius < 0: self.radius = 0
    
    def draw(self, screen):
        # Create a fading effect as particle dies
        alpha = int(255 * (self.lifespan / 40))
        if alpha > 0:
            surface = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(surface, self.color + (alpha,), (self.radius, self.radius), self.radius)
            screen.blit(surface, (self.x - self.radius, self.y - self.radius))

# --- Sound Generation ---
def generate_piano_wave(frequency, duration=0.6, sample_rate=44100, amplitude=0.4):
    n_samples = int(duration * sample_rate)
    t = np.linspace(0., duration, n_samples, endpoint=False)
    wave = np.sin(2 * np.pi * frequency * t)
    wave += 0.4 * np.sin(2 * np.pi * (2 * frequency) * t)
    wave += 0.2 * np.sin(2 * np.pi * (3 * frequency) * t)
    envelope = np.linspace(1, 0, n_samples)
    wave *= envelope
    wave = wave / np.max(np.abs(wave)) if np.max(np.abs(wave)) > 0 else wave
    sound_data = (wave * (2**15 - 1) * amplitude).astype(np.int16)
    stereo_sound_data = np.ascontiguousarray(np.vstack((sound_data, sound_data)).T)
    return pygame.sndarray.make_sound(stereo_sound_data)

# --- Finger Counting Logic ---
def count_fingers(hand_landmarks, hand_label):
    if not hand_landmarks: return 0
    tip_ids = [4, 8, 12, 16, 20]
    fingers_up = 0
    if hand_label == 'Right':
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 2].x: fingers_up += 1
    else: # Left
        if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 2].x: fingers_up += 1
    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y: fingers_up += 1
    return fingers_up

# --- Enhanced Drawing Functions ---
STARS = [(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT), random.randint(1, 2)) for _ in range(150)]
def draw_background(screen):
    screen.fill(BACKGROUND_COLOR)
    for x, y, r in STARS:
        # Twinkle effect by randomly changing brightness
        brightness = random.randint(100, 255)
        pygame.draw.circle(screen, (brightness, brightness, brightness), (x, y), r)

def draw_circular_webcam(screen, frame):
    frame_height, frame_width, _ = frame.shape
    cam_diameter = 300
    
    # Scale webcam frame and convert to Pygame surface
    frame = cv2.resize(frame, (cam_diameter, cam_diameter))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cam_surface = pygame.surfarray.make_surface(np.rot90(frame))
    
    # Create circular mask
    mask_surface = pygame.Surface((cam_diameter, cam_diameter), pygame.SRCALPHA)
    pygame.draw.circle(mask_surface, (255, 255, 255, 255), (cam_diameter // 2, cam_diameter // 2), cam_diameter // 2)
    
    # Apply mask to the webcam surface
    cam_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    
    # Position and draw the "magic mirror"
    pos_x = SCREEN_WIDTH // 2 - cam_diameter // 2
    pos_y = 150
    pygame.draw.circle(screen, KEY_PRESS_COLORS[4], (pos_x + cam_diameter//2, pos_y + cam_diameter//2), cam_diameter//2 + 8) # Border
    screen.blit(cam_surface, (pos_x, pos_y))

def draw_piano(screen, note_names, active_keys):
    num_keys = len(note_names)
    key_width = (SCREEN_WIDTH - 200) / num_keys
    key_height, key_start_x, key_start_y = 250, 100, SCREEN_HEIGHT - 350
    
    for i, name in enumerate(note_names):
        rect = pygame.Rect(key_start_x + i * (key_width + 5), key_start_y, key_width, key_height)
        shadow_rect, draw_rect = rect.move(0, 8), rect.copy()
        
        is_active = name in active_keys
        if not is_active:
            draw_rect.move_ip(0, -8)

        # Draw Shadow and Key for 3D effect
        pygame.draw.rect(screen, KEY_WHITE_SHADOW, shadow_rect, border_radius=15)
        final_color = KEY_PRESS_COLORS[i % 5] if is_active else KEY_WHITE_COLOR
        pygame.draw.rect(screen, final_color, draw_rect, border_radius=15)
        pygame.draw.rect(screen, BLACK_COLOR, draw_rect, 3, border_radius=15) # Border
        
        key_font = pygame.font.Font(None, 32)
        note_text = key_font.render(name, True, BLACK_COLOR)
        screen.blit(note_text, (rect.centerx - note_text.get_width() / 2, rect.bottom - 45))
        
# --- Main Application ---
def run_piano_app():
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=4, buffer=512)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Magical Hand Piano")
    title_font = pygame.font.Font(None, 80)
    info_font = pygame.font.Font(None, 40)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Error: Could not open webcam."); return

    notes_freq = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 587.33, 659.25]
    note_names = ["C", "D", "E", "F", "G", "A", "B", "C+", "D+", "E+"]
    piano_sounds = [generate_piano_wave(freq) for freq in notes_freq]

    # State variables including debounce logic
    stable_fingers = {'Left': 0, 'Right': 0}
    pending_fingers = {'Left': 0, 'Right': 0}
    debounce_counters = {'Left': 0, 'Right': 0}

    # <<< FIX: Declare that we are using the global PARTICLES list
    global PARTICLES

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
        
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        raw_fingers = {'Left': 0, 'Right': 0}
        active_keys = set()
        
        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = handedness.classification[0].label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                raw_fingers[hand_label] = count_fingers(hand_landmarks, hand_label)

        for hand_label in ['Left', 'Right']:
            # Debounce logic
            if raw_fingers[hand_label] == pending_fingers[hand_label]:
                debounce_counters[hand_label] += 1
            else:
                pending_fingers[hand_label] = raw_fingers[hand_label]
                debounce_counters[hand_label] = 0

            if debounce_counters[hand_label] >= DEBOUNCE_FRAMES:
                if stable_fingers[hand_label] != pending_fingers[hand_label]:
                    stable_fingers[hand_label] = pending_fingers[hand_label]
                    
                    if stable_fingers[hand_label] > 0:
                        offset = 0 if hand_label == 'Left' else 5
                        note_idx = offset + stable_fingers[hand_label] - 1
                        if 0 <= note_idx < len(piano_sounds):
                            piano_sounds[note_idx].play()
                            key_x_pos = 100 + note_idx * ((SCREEN_WIDTH - 200) / 10 + 5) + 50
                            for _ in range(30): PARTICLES.append(Particle(key_x_pos, SCREEN_HEIGHT - 350, KEY_PRESS_COLORS[note_idx % 5]))

            # Update active key visuals based on stable gesture
            if stable_fingers[hand_label] > 0:
                offset = 0 if hand_label == 'Left' else 5
                note_idx = offset + stable_fingers[hand_label] - 1
                if 0 <= note_idx < len(note_names): active_keys.add(note_names[note_idx])

        # --- Drawing on Pygame Screen ---
        draw_background(screen)
        draw_piano(screen, note_names, active_keys)

        # Update and draw particles
        for p in PARTICLES: p.update()
        PARTICLES = [p for p in PARTICLES if p.lifespan > 0]
        for p in PARTICLES: p.draw(screen)

        draw_circular_webcam(screen, frame)
        
        # UI Text 
        title_text = title_font.render("Magical Hand Piano", True, WHITE_COLOR)
        info_l = info_font.render(f"Left: {stable_fingers['Left']}", True, WHITE_COLOR)
        info_r = info_font.render(f"Right: {stable_fingers['Right']}", True, WHITE_COLOR)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() / 2, 40))
        screen.blit(info_l, (100, 60))
        screen.blit(info_r, (SCREEN_WIDTH - 100 - info_r.get_width(), 60))
        
        pygame.display.flip()

    cap.release()
    pygame.quit()
    print("Application closed successfully.")

if __name__ == '__main__':
    run_piano_app()
