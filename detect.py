import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk

# Set parameters
frame_height = 64
frame_width = 64
frame_count = 30  # Number of frames per sequence
display_height = 320  # New display height for the enlarged video preview
display_width = 320   # New display width for the enlarged video preview

# Load the trained model
model = load_model('model.h5')

# Flag to stop live camera feed
camera_running = False

# Function to preprocess frames for the model
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (frame_width, frame_height))
    frame_normalized = frame_resized / 255.0  # Normalize pixel values to [0, 1]
    return frame_normalized

# Function to process and predict from a video file
def process_video(video_path, label_display):
    video = cv2.VideoCapture(video_path)
    frames = []  # List to store frames for prediction

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Preprocess the frame for prediction
        frame_normalized = preprocess_frame(frame)
        frames.append(frame_normalized)

        # Once we have 30 frames, predict
        if len(frames) == frame_count:
            # Convert the list of frames to a numpy array
            frames_array = np.array(frames)
            frames_array = np.expand_dims(frames_array, axis=0)  # Add batch dimension (1, 30, 64, 64, 3)

            # Predict
            prediction = model.predict(frames_array)
            label = np.argmax(prediction, axis=1)[0]

            # Display the result
            if label == 0:
                label_display.config(text="Prediction: Normal", fg="green", font=("Arial", 16, "bold"))
            else:
                label_display.config(text="Prediction: Shoplifting", fg="red", font=("Arial", 16, "bold"))

            # Reset the frames list for the next sequence
            frames = []

        # Display the current frame in the Tkinter window
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)

        # Resize the frame for display purposes (keeping the aspect ratio)
        frame_image_resized = frame_image.resize((display_width, display_height), Image.Resampling.LANCZOS)

        frame_image_tk = ImageTk.PhotoImage(frame_image_resized)
        
        # Keep a reference to the image for Tkinter to display it
        canvas.image = frame_image_tk  # Retain the image reference
        canvas.create_image(0, 0, image=frame_image_tk, anchor=tk.NW)
        window.update_idletasks()
        window.update()

    video.release()

# Function to start live camera feed
def start_live_camera(label_display):
    global camera_running
    global frames  # Declare frames as global
    camera_running = True
    frames = []  # Initialize frames list for live camera

    def update_frame():
        global frames  # Declare frames as global
        if not camera_running:
            cap.release()  # Release the camera when stopping
            return
        
        ret, frame = cap.read()
        if not ret:
            return

        # Preprocess the frame for prediction
        frame_normalized = preprocess_frame(frame)
        frames.append(frame_normalized)

        # Once we have 30 frames, predict
        if len(frames) == frame_count:
            # Convert the list of frames to a numpy array
            frames_array = np.array(frames)
            frames_array = np.expand_dims(frames_array, axis=0)  # Add batch dimension (1, 30, 64, 64, 3)

            # Predict
            prediction = model.predict(frames_array)
            label = np.argmax(prediction, axis=1)[0]

            # Display the result
            if label == 0:
                label_display.config(text="Prediction: Normal", fg="green", font=("Arial", 16, "bold"))
            else:
                label_display.config(text="Prediction: Shoplifting", fg="red", font=("Arial", 16, "bold"))

            # Reset the frames list for the next sequence
            frames = []

        # Display the current frame in the Tkinter window
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)

        # Resize the frame for display purposes (keeping the aspect ratio)
        frame_image_resized = frame_image.resize((display_width, display_height), Image.Resampling.LANCZOS)

        frame_image_tk = ImageTk.PhotoImage(frame_image_resized)
        
        # Keep a reference to the image for Tkinter to display it
        canvas.image = frame_image_tk  # Retain the image reference
        canvas.create_image(0, 0, image=frame_image_tk, anchor=tk.NW)

        # Schedule the next frame update
        window.after(10, update_frame)  # Schedule the next frame update

    cap = cv2.VideoCapture(0)
    update_frame()  # Start updating the frames from the live camera

# Function to stop live camera feed
def stop_live_camera():
    global camera_running
    camera_running = False

# Function to upload video and start processing
def upload_video(label_display):
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("MP4 Files", "*.mp4")])
    if video_path:
        process_video(video_path, label_display)

# Setting up the Tkinter GUI
window = tk.Tk()
window.title("Shoplifting Detection")

# Enlarged window size
window.geometry("1200x800")  # Resize the window (increased size)

# Set a background color
window.config(bg="#2C3E50")  # Dark background color for a sleek look

# Add a header label
header_label = tk.Label(window, text="Shoplifting Detection", font=("Arial", 24, "bold"), fg="white", bg="#2C3E50")
header_label.pack(pady=20)

# Canvas to display the video frames (increased size for larger preview)
canvas = tk.Canvas(window, width=display_width, height=display_height, bg="black")  # Enlarged canvas
canvas.pack(pady=10)

# Label to display predictions
label_display = tk.Label(window, text="Prediction: Waiting for Input", font=("Arial", 16), fg="white", bg="#2C3E50")
label_display.pack(pady=20)

# Buttons to upload video or start live camera feed
btn_upload_video = tk.Button(window, text="Upload Video", command=lambda: upload_video(label_display),
                             font=("Arial", 14), fg="white", bg="#27AE60", width=20, height=2, bd=0, relief="solid")
btn_upload_video.pack(pady=10)

btn_live_camera = tk.Button(window, text="Start Live Camera Feed", command=lambda: start_live_camera(label_display),
                            font=("Arial", 14), fg="white", bg="#FF5733", width=20, height=2, bd=0, relief="solid")
btn_live_camera.pack(pady=10)

btn_stop_camera = tk.Button(window, text="Stop Live Camera", command=stop_live_camera,
                            font=("Arial", 14), fg="white", bg="#C0392B", width=20, height=2, bd=0, relief="solid")
btn_stop_camera.pack(pady=10)

# Add a footer label with credits or extra info
footer_label = tk.Label(window, text="Developed by Smile Trackers", font=("Arial", 10), fg="white", bg="#2C3E50")
footer_label.pack(side=tk.BOTTOM, pady=10)

# Run the Tkinter main loop
window.mainloop()
