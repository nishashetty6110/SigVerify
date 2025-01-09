import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
from PIL import Image, ImageTk
import cv2
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim

class SignatureApp:
    def __init__(self):
        self.window = tk.Tk()
        # self.window.attributes('-fullscreen', True)

        self.window.title("SigVerify App")
        self.window.geometry("2000x1000")

        # Directory for storing signatures
        self.signature_dir = "./signatures"
        os.makedirs(self.signature_dir, exist_ok=True)

        # Load the trained signature matching model
        self.model = tf.keras.models.load_model('./models/model_cnn.h5')
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Set up navigation
        self.nav_frame = tk.Frame(self.window, bg="#003366")
        self.nav_frame.pack(fill=tk.X)

        # Navigation buttons
        self.create_nav_buttons()

        # Main content area for pages
        self.content_frame = tk.Frame(self.window, bg="white")
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        # Create pages
        self.home_page = self.create_home_page()
        self.about_page = self.create_about_page()
        self.add_signature_page = self.create_add_signature_page()
        self.verify_signature_page = self.create_verify_signature_page()

        # Display Home page by default
        self.show_home()

    def create_nav_buttons(self):
        nav_buttons = [
            ("Home", self.show_home),
            ("About Us", self.show_about),
            ("Add Signature", self.show_add_signature),
            ("Verify Signature", self.show_verify_signature),
        ]
        for text, command in nav_buttons:
            tk.Button(
                self.nav_frame,
                text=text,
                command=command,
                bg="white",
                fg="#003366",
                font=("Arial", 25, "bold")
            ).pack(side=tk.LEFT, padx=60, pady=10)

    def create_home_page(self):
        canvas = tk.Canvas(self.content_frame, highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        # Load and display the background image
        try:
            bg_image = Image.open("signbg.png")
            bg_image = bg_image.resize((1670,700), Image.Resampling.LANCZOS)
            bg_photo = ImageTk.PhotoImage(bg_image)
            canvas.create_image(0, 0, image=bg_photo, anchor="nw")
            canvas.image = bg_photo
        except Exception as e:
            print("Error loading background image:", e)

        return canvas
    def create_about_page(self):
        canvas = tk.Canvas(self.content_frame, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        self.video_source = "aboutus.mp4"  # Path to your video file
        self.capture = cv2.VideoCapture(self.video_source)

        # Call the function to start the video loop
        self.play_video(canvas)
        return canvas
    def play_video(self, canvas):
        # Read the next frame from the video
        ret, frame = self.capture.read()

        if ret:
            # Convert the frame to RGB (from BGR used by OpenCV)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to a PIL image
            image = Image.fromarray(frame)
            image = image.resize((1550, 750), Image.Resampling.LANCZOS)  # Resize to fit the window
            photo = ImageTk.PhotoImage(image)

            # Display the image on the canvas
            canvas.create_image(0, 0, image=photo, anchor="nw")
            canvas.image = photo  # Keep a reference to the image to prevent garbage collection
            self.window.after(20, self.play_video, canvas)  # Call this function again after 10ms to create the loop
        else:
            # If video ends, start again from the beginning
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.play_video(canvas)


    def create_add_signature_page(self):
        canvas = tk.Canvas(self.content_frame, highlightthickness=0)
        canvas.pack(fill="both", expand=True)

    # Load and display the background image
        try:
            bg_image = Image.open("addsignbg.png")  # Background image path
            bg_image = bg_image.resize((2000, 900), Image.Resampling.LANCZOS)
            bg_photo = ImageTk.PhotoImage(bg_image)
            canvas.create_image(0, 0, image=bg_photo, anchor="nw")
            canvas.image = bg_photo  # Keep a reference to avoid garbage collection
        except Exception as e:
            print("Error loading background image:", e)
        tk.Label(canvas, text="Enter Full Name:", bg="white", font=("Arial", 16)).pack(pady=10)
        self.name_entry = tk.Entry(canvas, width=30, font=("Arial", 16))
        self.name_entry.pack(pady=5)
        tk.Label(canvas, text="Upload or Capture Signature:", bg="white", font=("Arial", 16)).pack(pady=10)
        self.file_path_label = tk.Label(canvas, text="", bg="white", font=("Arial", 14), fg="green")
        self.file_path_label.pack(pady=5)

        tk.Button(canvas, text="Upload Signature", command=self.upload_signature, bg="#4A90E2", fg="white", font=("Arial", 20)).pack(pady=15)
        tk.Button(canvas, text="Capture Signature", command=self.capture_signature, bg="#4A90E2", fg="white", font=("Arial", 20)).pack(pady=5)
        tk.Button(canvas, text="Add", command=self.add_signature, bg="#34A853", fg="white", font=("Arial", 20)).pack(pady=20)
        return canvas

    def create_verify_signature_page(self):
        # Create a canvas to hold the background image
        self.canvas = tk.Canvas(self.content_frame, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Load and display the background image
        try:
            bg_image = Image.open("verifysigbg.png")  # Background image path
            bg_image = bg_image.resize((2000, 1000), Image.Resampling.LANCZOS)
            bg_photo = ImageTk.PhotoImage(bg_image)
            self.canvas.create_image(0, 0, image=bg_photo, anchor="nw")
            self.canvas.image = bg_photo  # Keep a reference to avoid garbage collection
        except Exception as e:
            print("Error loading background image:", e)

        tk.Label(self.canvas, text="Upload or Capture Signature to Verify:", bg="white", font=("Arial", 26)).pack(pady=60)

        # Add buttons for upload and capture options
        tk.Button(self.canvas, text="Upload Signature", command=self.upload_for_verification, bg="#34A853", fg="white", font=("Arial", 25)).pack(pady=10)
        tk.Button(self.canvas, text="Capture Signature", command=self.capture_for_verification, bg="#4A90E2", fg="white", font=("Arial", 25)).pack(pady=10)

        self.verify_result_label = tk.Label(self.canvas, text="", font=("Arial", 20))
        self.verify_result_label.pack(pady=20)

        # Add a placeholder for the matched signature image
        self.matched_signature_label = None

        return self.canvas



    def upload_signature(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;")])
        if not self.name_entry.get().strip():
            messagebox.showwarning("Warning", "Please enter a name.")
            return
        if self.file_path:
        # Get the file name from the file path
            file_name = os.path.basename(self.file_path)
            
            # Update the label text with the file name
            self.file_path_label.config(text=f"Selected file: {file_name}")

    def capture_signature(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter a name.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to read from camera")
                break
            cv2.imshow("Capture Signature - Press 'C' to capture, 'Q' to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                self.file_path = os.path.join(self.signature_dir, f"{name.replace(' ', '_')}.jpg")
                cv2.imwrite(self.file_path, frame)
                messagebox.showinfo("Capture Successful", "Signature captured successfully.")
                break
            elif key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def add_signature(self):
        name = self.name_entry.get().strip()
        if not name or not hasattr(self, 'file_path') or not self.file_path:
            messagebox.showwarning("Warning", "Please fill in all fields.")
            return
        
        # Compare with existing signatures
        for filename in os.listdir(self.signature_dir):
            existing_path = os.path.join(self.signature_dir, filename)
            if self.file_path != existing_path and self.compare_signatures(self.file_path, existing_path):
                messagebox.showerror("Error", "This signature already exists.")
                return
        
        # Save the new signature
        try:
            file_name = f"{name.replace(' ', '_')}.jpg"
            Image.open(self.file_path).save(os.path.join(self.signature_dir, file_name))
            messagebox.showinfo("Success", "Signature added successfully.")
            self.name_entry.delete(0, 'end')  # Clear the name entry field
            self.file_path = None  # Reset the file path
            self.file_path_label.config(text="")  # Clear the file path label
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add signature: {e}")

    def capture_for_verification(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to read from camera")
                break
            
            # Display the video feed
            cv2.imshow("Capture Signature - Press 'C' to capture, 'Q' to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                # Save the captured frame temporarily
                self.file_path = "./captured_signature.jpg"
                
                # Process the frame to enhance quality
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)  # Binarization
                
                # Save the processed image
                cv2.imwrite(self.file_path, binary)
                
                messagebox.showinfo("Capture Successful", "Signature captured successfully.")
                
                # Proceed with verification
                self.verify_signature(self.file_path)
                break
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def verify_signature(self, file_path):
        if hasattr(self, 'matched_signature_label') and self.matched_signature_label:
            self.canvas.delete(self.matched_signature_label)
        if hasattr(self, 'verify_result_label'):
            self.verify_result_label.config(text="")  # Clear previous result

        # Preprocess the uploaded or captured signature
        uploaded_image = self.preprocess_image(file_path)
        
        # Make the prediction with the model
        prediction = self.model.predict(uploaded_image)

        # Adjust the threshold to a lower value if needed
        threshold = 0.7  # You can experiment with a lower threshold value (like 0.4 or 0.3)
        
        if prediction[0][0] > threshold:  # Adjust the threshold based on testing
            matched_filename = self.find_matching_signature(file_path)
            if matched_filename:
                name = os.path.splitext(os.path.basename(matched_filename))[0].replace('_', ' ')
                self.verify_result_label.config(text=f"Signature matched successfully! \nVerified as: {name}")
                        
                # Load and display the matched signature image
                try:
                    matched_image = Image.open(matched_filename)  # Image of the matched signature
                    matched_image = matched_image.resize((200, 100), Image.Resampling.LANCZOS)  # Resize if necessary
                    matched_image_photo = ImageTk.PhotoImage(matched_image)

                    # If there was a previous matched image, remove it
                    if self.matched_signature_label:
                        self.canvas.delete(self.matched_signature_label)

                    # Create the image below the result label
                    self.matched_signature_label = self.canvas.create_image(790, 500, image=matched_image_photo)

                    # Keep reference to avoid garbage collection
                    self.canvas.matched_image_photo = matched_image_photo

                except Exception as e:
                    print("Error displaying matched signature image:", e)
            else:
                self.verify_result_label.config(text="Signature not matched.")
        else:
            self.verify_result_label.config(text="Signature not matched.")

    def upload_for_verification(self):
        # Open file dialog to upload a signature
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return  # If no file is selected, do nothing

        # Call the verify method with the uploaded file
        self.verify_signature(file_path)  # Directly call the verify_signature method


    def preprocess_image(self, file_path):
        img = Image.open(file_path).resize((150, 150)).convert('RGB')  # Ensure RGB format
        img_array = np.array(img) / 255.0  # Normalize pixel values
        return img_array.reshape(-1, 150, 150, 3)  # Reshape with 3 channels for the model


    def compare_signatures(self, uploaded_file_path, stored_file_path):
        try:
            uploaded_img = Image.open(uploaded_file_path).convert("L").resize((150, 150))
            stored_img = Image.open(stored_file_path).convert("L").resize((150, 150))
            uploaded_array = np.array(uploaded_img)
            stored_array = np.array(stored_img)
            
            # Compute SSIM
            similarity_index, _ = ssim(uploaded_array, stored_array, full=True)
            print(f"SSIM between {uploaded_file_path} and {stored_file_path}: {similarity_index}")
            
            return similarity_index > 0.85  # Threshold for matching
        except Exception as e:
            print(f"Error comparing signatures: {e}")
            return False


    
    def find_matching_signature(self, uploaded_file_path):
        # Loop through all files in the signature directory
        for filename in os.listdir(self.signature_dir):
            signature_path = os.path.join(self.signature_dir, filename)
            if os.path.isfile(signature_path):
                # Compare the uploaded signature with the stored signature
                if self.compare_signatures(uploaded_file_path, signature_path):
                    return signature_path
        return None
    def show_home(self):
        self.clear_pages()
        self.home_page.pack(fill=tk.BOTH, expand=True)

    def show_about(self):
        self.clear_pages()
        self.about_page.pack(fill=tk.BOTH, expand=True)


    def show_add_signature(self):
        self.clear_pages()
        self.add_signature_page.pack(fill=tk.BOTH, expand=True)

    def show_verify_signature(self):
        self.clear_pages()
        self.verify_signature_page.pack(fill=tk.BOTH, expand=True)

    def clear_pages(self):
        if hasattr(self, 'verify_result_label'):
            self.verify_result_label.config(text="")  # Clear result text
            if hasattr(self, 'matched_signature_label') and self.matched_signature_label:
                self.canvas.delete(self.matched_signature_label)  
        for page in (self.home_page, self.about_page, self.add_signature_page, self.verify_signature_page):
            page.pack_forget()

    def run(self):
        self.window.mainloop()


