import cv2
import mediapipe as mp
import random
import time
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class FallingBall:
    def __init__(self, screen_width, screen_height, ball_type='green'):
        self.x = random.randint(50, screen_width - 50)
        self.y = -30  # Start from top
        self.radius = 25
        self.speed = random.randint(6, 10)  # Faster speed
        self.type = ball_type  # 'green', 'red', or 'yellow'
        self.alive = True
        
        # Set color based on type
        if ball_type == 'green':
            self.color = (0, 255, 0)  # Green
        elif ball_type == 'red':
            self.color = (0, 0, 255)  # Red
        elif ball_type == 'yellow':
            self.color = (0, 255, 255)  # Yellow
            
    def update(self):
        """Update ball position (falling down)"""
        if self.alive:
            self.y += self.speed
            
    def draw(self, frame):
        if self.alive:
            # Draw ball with glow effect
            cv2.circle(frame, (self.x, self.y), self.radius + 5, self.color, 2)
            cv2.circle(frame, (self.x, self.y), self.radius, self.color, -1)
            
    def check_collision(self, hand_x, hand_y):
        """Check if hand touches the ball"""
        if self.alive:
            distance = np.sqrt((self.x - hand_x)**2 + (self.y - hand_y)**2)
            if distance < self.radius + 20:
                self.alive = False
                return True
        return False
    
    def is_out_of_screen(self, screen_height):
        """Check if ball fell off screen"""
        return self.y > screen_height + 50

class MediaPipeGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.score = 0
        self.lives = 3
        self.balls = []
        self.game_over = False
        self.start_time = time.time()
        self.last_spawn_time = time.time()
        self.spawn_interval = 0.8  # Spawn ball every 0.8 second (faster)
        
        # Power-up mode
        self.powerup_mode = False
        self.powerup_start_time = 0
        self.powerup_duration = 10  # 10 seconds
        
        # Finger trail effect
        self.finger_trail = []  # Store last positions for trail effect
        self.max_trail_length = 15  # Number of trail points
        
        # Get camera dimensions
        ret, frame = self.cap.read()
        if ret:
            self.height, self.width = frame.shape[:2]
        else:
            self.width, self.height = 640, 480
            
        # Initialize hands only (NO FACE MESH)
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
    
    def spawn_ball(self):
        """Spawn a new falling ball"""
        current_time = time.time()
        
        if current_time - self.last_spawn_time >= self.spawn_interval:
            if self.powerup_mode:
                # During power-up: all balls are yellow
                ball_type = 'yellow'
            else:
                # Normal mode: 70% green, 30% red
                ball_type = 'green' if random.random() < 0.7 else 'red'
            
            self.balls.append(FallingBall(self.width, self.height, ball_type))
            self.last_spawn_time = current_time
    
    def activate_powerup(self):
        """Activate power-up mode"""
        self.powerup_mode = True
        self.powerup_start_time = time.time()
        # Convert all existing balls to yellow
        for ball in self.balls:
            if ball.alive:
                ball.type = 'yellow'
                ball.color = (0, 255, 255)
    
    def check_powerup_timer(self):
        """Check if power-up mode should end"""
        if self.powerup_mode:
            elapsed = time.time() - self.powerup_start_time
            if elapsed >= self.powerup_duration:
                self.powerup_mode = False
    
    def draw_ui(self, frame):
        """Draw game UI - CLEANED VERSION"""
        # Semi-transparent overlay
        overlay = frame.copy()
        
        # Score display (top left)
        cv2.rectangle(overlay, (10, 10), (220, 70), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"SCORE: {self.score}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        
        # Lives display (top right - DRAW HEARTS)
        lives_x_start = self.width - 180
        for i in range(self.lives):
            heart_x = lives_x_start + (i * 55)
            heart_y = 35
            # Draw heart shape using circles and polygon
            cv2.circle(frame, (heart_x - 10, heart_y), 15, (0, 0, 255), -1)
            cv2.circle(frame, (heart_x + 10, heart_y), 15, (0, 0, 255), -1)
            pts = np.array([[heart_x - 25, heart_y], [heart_x, heart_y + 30], 
                           [heart_x + 25, heart_y]], np.int32)
            cv2.fillPoly(frame, [pts], (0, 0, 255))
            # White outline
            cv2.circle(frame, (heart_x - 10, heart_y), 15, (255, 255, 255), 2)
            cv2.circle(frame, (heart_x + 10, heart_y), 15, (255, 255, 255), 2)
            cv2.polylines(frame, [pts], True, (255, 255, 255), 2)
        
        # Timer
        elapsed_time = int(time.time() - self.start_time)
        cv2.putText(frame, f"Time: {elapsed_time}s", (self.width // 2 - 70, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Power-up timer (if active)
        if self.powerup_mode:
            remaining = int(self.powerup_duration - (time.time() - self.powerup_start_time))
            if remaining > 0:
                # Flashing effect
                if int(time.time() * 2) % 2 == 0:
                    color = (0, 255, 255)
                else:
                    color = (0, 200, 200)
                    
                cv2.putText(frame, f"POWER-UP! {remaining}s", 
                            (self.width // 2 - 120, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Title at bottom (centered)
        cv2.putText(frame, "MediaPipe Game", (self.width // 2 - 150, self.height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Game Over screen
        if self.game_over:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, "GAME OVER!", (self.width // 2 - 180, self.height // 2 - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 4)
            cv2.putText(frame, f"Final Score: {self.score}", (self.width // 2 - 150, self.height // 2 + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(frame, "Press 'R' to Restart or 'Q' to Quit", 
                        (self.width // 2 - 280, self.height // 2 + 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def reset_game(self):
        """Reset game to initial state"""
        self.score = 0
        self.lives = 3
        self.balls = []
        self.game_over = False
        self.start_time = time.time()
        self.last_spawn_time = time.time()
        self.powerup_mode = False
        self.finger_trail = []  # Clear trail
    
    def run(self):
        print("\n" + "="*50)
        print("    MEDIAPIPE FALLING BALLS GAME")
        print("="*50)
        print("\nInstructions:")
        print("🟢 GREEN balls   = +10 points (MUST catch or lose life!)")
        print("🔴 RED balls     = -1 life (avoid them!)")
        print("🟡 YELLOW balls  = +10 points (bonus mode!)")
        print("\n⚠️  Miss a green ball = -1 life!")
        print("💡 Power-up activates every 100 points!")
        print("   All balls turn yellow for 10 seconds!")
        print("\n⌨️  Controls:")
        print("   Q = Quit game")
        print("   R = Restart (after game over)")
        print("\n" + "="*50)
        print("Starting game...\n")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if not self.game_over:
                # Process hands only (NO FACE MESH - face is clear)
                hand_results = self.hands.process(rgb_frame)
                
                # Draw hands and check collisions
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                        
                        # Get index finger tip position
                        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        hand_x = int(index_finger_tip.x * self.width)
                        hand_y = int(index_finger_tip.y * self.height)
                        
                        # Add current position to trail
                        self.finger_trail.append((hand_x, hand_y))
                        if len(self.finger_trail) > self.max_trail_length:
                            self.finger_trail.pop(0)
                        
                        # Draw trail effect (lines connecting previous positions)
                        for i in range(1, len(self.finger_trail)):
                            # Calculate thickness and color based on position in trail
                            thickness = int((i / len(self.finger_trail)) * 8) + 1
                            alpha = int((i / len(self.finger_trail)) * 255)
                            color = (0, 255, 0)  # Green trail
                            
                            cv2.line(frame, self.finger_trail[i-1], self.finger_trail[i], 
                                   color, thickness)
                        
                        # Draw finger tip indicator (larger during power-up)
                        indicator_size = 20 if not self.powerup_mode else 25
                        cv2.circle(frame, (hand_x, hand_y), indicator_size, (0, 255, 0), -1)
                        cv2.circle(frame, (hand_x, hand_y), indicator_size + 5, (255, 255, 255), 2)
                        
                        # Check collision with balls
                        for ball in self.balls:
                            if ball.check_collision(hand_x, hand_y):
                                if ball.type == 'green' or ball.type == 'yellow':
                                    self.score += 10
                                    # Check if score reached 100 multiple (activate power-up)
                                    if self.score > 0 and self.score % 100 == 0 and not self.powerup_mode:
                                        self.activate_powerup()
                                elif ball.type == 'red':
                                    self.lives -= 1
                                    if self.lives <= 0:
                                        self.game_over = True
                
                # Spawn new balls
                self.spawn_ball()
                
                # Update and draw balls
                for ball in self.balls[:]:
                    ball.update()
                    ball.draw(frame)
                    
                    # Check if ball fell off screen
                    if ball.is_out_of_screen(self.height):
                        # If green ball is missed, lose a life
                        if ball.alive and ball.type == 'green':
                            self.lives -= 1
                            if self.lives <= 0:
                                self.game_over = True
                        self.balls.remove(ball)
                
                # Check power-up timer
                self.check_powerup_timer()
            
            # Draw UI
            self.draw_ui(frame)
            
            # Display frame
            cv2.imshow('MediaPipe Falling Balls Game', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and self.game_over:
                self.reset_game()
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    game = MediaPipeGame()
    game.run()
