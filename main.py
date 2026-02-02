# =============================================================================
# PROJECT: AI Student Safety & Uniform Monitor (PRODUCTION READY)
# TECH STACK: Python, OpenCV, YOLOv8, Tkinter, Pandas, Matplotlib
# =============================================================================

import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg") # Set backend for Tkinter compatibility

from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
LOG_FILE = "logs/entry_logs.csv"
MODEL_PATH = "yolov8n.pt"

# Dress Code Logic
LOWER_BLUE = np.array([90, 50, 50])
UPPER_BLUE = np.array([130, 255, 255])
LOWER_WHITE = np.array([0, 0, 200])
UPPER_WHITE = np.array([180, 30, 255])

# UI Colors
COLOR_BG = "#2C3E50"
COLOR_ACCENT = "#3498DB"
COLOR_SUCCESS = "#2ECC71"
COLOR_DANGER = "#E74C3C"
COLOR_TEXT = "#ECF0F1"
COLOR_WARN = "#F1C40F"

# =============================================================================
# LOGIC MODULE: LOGGER
# =============================================================================
class DataLogger:
    def __init__(self):
        self.init_csv()

    def init_csv(self):
        if not os.path.exists("logs"):
            os.makedirs("logs")
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Date", "Time", "Status", "Confidence", "Notes"])

    def log_entry(self, status, confidence, notes=""):
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([date_str, time_str, status, f"{confidence:.2f}", notes])
        return date_str, time_str

# =============================================================================
# LOGIC MODULE: AI DETECTOR
# =============================================================================
class AIDetector:
    def __init__(self):
        print("[System] Loading YOLO Model...")
        self.model = YOLO(MODEL_PATH)
        self.class_names = self.model.names

    def process_frame(self, frame):
        """
        Runs YOLO inference and analyzes dress code compliance.
        """
        results = self.model(frame, verbose=False)
        detected_status = "Monitoring..."
        current_max_conf = 0.0
        boxes_found = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id == 0 and conf > 0.5: 
                    boxes_found = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    current_max_conf = max(current_max_conf, conf)
                    
                    # ROI: Top half (Upper Body)
                    h, w, _ = frame.shape
                    roi_y_start = max(0, y1)
                    roi_y_end = max(0, y1 + (y2 - y1) // 2) 
                    roi_x_start = max(0, x1)
                    roi_x_end = min(w, x2)
                    
                    roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                    
                    if roi.size > 0:
                        is_compliant = self.check_uniform_color(roi)
                        
                        if is_compliant:
                            status = "Allowed"
                            color = (0, 255, 0)
                            detected_status = "Allowed"
                        else:
                            status = "Violation"
                            color = (0, 0, 255)
                            detected_status = "Violation"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{status} {conf:.0%}"
                        t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
                        c2 = x1 + t_size[0], y1 - t_size[1] - 3
                        cv2.rectangle(frame, (x1, y1), c2, color, -1)
                        cv2.putText(frame, label, (x1, y1 - 2), 0, 0.60, [255, 255, 255], thickness=1)

        return frame, detected_status, current_max_conf

    def check_uniform_color(self, roi):
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv_roi, LOWER_BLUE, UPPER_BLUE)
        mask_white = cv2.inRange(hsv_roi, LOWER_WHITE, UPPER_WHITE)
        
        blue_pixels = cv2.countNonZero(mask_blue)
        white_pixels = cv2.countNonZero(mask_white)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        threshold = total_pixels * 0.10
        return (blue_pixels + white_pixels) > threshold

# =============================================================================
# MODULE: ADMIN PANEL (IMPROVED)
# =============================================================================
class AdminPanel:
    def __init__(self, parent):
        self.window = Toplevel(parent)
        self.window.title("Admin Panel - Student Analytics")
        self.window.geometry("1100x750")
        self.window.configure(bg=COLOR_BG)
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.current_filter = "Today" # Default view
        self.data = self.load_logs()
        
        self.setup_ui()
        self.update_dashboard()

    def load_logs(self):
        if not os.path.exists(LOG_FILE):
            return pd.DataFrame(columns=["Date", "Time", "Status", "Confidence", "Notes"])
        
        try:
            df = pd.read_csv(LOG_FILE)
            
            # Performance Improvement: Filter by date if selected
            if self.current_filter == "Today":
                today_str = datetime.now().strftime("%Y-%m-%d")
                df = df[df['Date'] == today_str]
            return df
        except Exception as e:
            print(f"Error loading logs: {e}")
            return pd.DataFrame()

    def setup_ui(self):
        # Header
        header = Frame(self.window, bg="#1C2833", height=60)
        header.pack(fill=X)
        Label(header, text="ADMIN DASHBOARD", font=("Helvetica", 18, "bold"), 
              bg="#1C2833", fg="white").pack(pady=15)

        # Filter Controls
        control_bar = Frame(self.window, bg=COLOR_BG)
        control_bar.pack(fill=X, padx=20, pady=5)
        
        Button(control_bar, text="View Today's Logs", command=lambda: self.change_filter("Today"), 
               bg=COLOR_ACCENT, fg="white", font=("bold", 10)).pack(side=LEFT, padx=5)
        Button(control_bar, text="View All History", command=lambda: self.change_filter("All"), 
               bg="#7F8C8D", fg="white", font=("bold", 10)).pack(side=LEFT, padx=5)
        
        # Stats Row
        stats_row = Frame(self.window, bg=COLOR_BG)
        stats_row.pack(fill=X, padx=20, pady=10)

        self.card_total = self.create_stat_card(stats_row, "TOTAL SCANS", "0", COLOR_ACCENT)
        self.card_total.pack(side=LEFT, padx=10, fill=Y)

        self.card_rate = self.create_stat_card(stats_row, "COMPLIANCE RATE", "0%", COLOR_SUCCESS)
        self.card_rate.pack(side=LEFT, padx=10, fill=Y)

        self.card_violations = self.create_stat_card(stats_row, "VIOLATIONS", "0", COLOR_DANGER)
        self.card_violations.pack(side=LEFT, padx=10, fill=Y)

        # Charts
        chart_frame = Frame(self.window, bg=COLOR_BG, height=300)
        chart_frame.pack(fill=X, padx=20, pady=10)
        chart_frame.pack_propagate(False)
        self.canvas_frame = Frame(chart_frame, bg="#34495E")
        self.canvas_frame.pack(fill=BOTH, expand=True)

        # Logs Table
        table_frame = Frame(self.window, bg=COLOR_BG)
        table_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)

        cols = ("Date", "Time", "Status", "Confidence", "Notes")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=10)
        
        for col in cols:
            self.tree.heading(col, text=col)
        self.tree.column("Date", width=100)
        self.tree.column("Time", width=80)
        self.tree.column("Status", width=100)
        self.tree.column("Confidence", width=80)
        self.tree.column("Notes", width=300)

        scrollbar = ttk.Scrollbar(table_frame, orient=VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        self.tree.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

    def create_stat_card(self, parent, title, value, color):
        frame = Frame(parent, bg="#34495E", width=200, height=80, padx=15, pady=10)
        Label(frame, text=title, font=("Helvetica", 10), bg="#34495E", fg="#BDC3C7").pack(anchor=W)
        lbl_val = Label(frame, text=value, font=("Helvetica", 24, "bold"), bg="#34495E", fg=color)
        lbl_val.pack(anchor=W)
        return lbl_val

    def change_filter(self, mode):
        self.current_filter = mode
        self.data = self.load_logs()
        self.update_dashboard()

    def update_dashboard(self):
        # Update Text
        total = len(self.data)
        violations = len(self.data[self.data['Status'] == 'Violation'])
        allowed = len(self.data[self.data['Status'] == 'Allowed'])
        
        self.card_total.config(text=str(total))
        self.card_violations.config(text=str(violations))
        
        rate = int((allowed / total) * 100) if total > 0 else 0
        self.card_rate.config(text=f"{rate}%")

        # Update Chart
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
        fig.patch.set_facecolor('#34495E') 
        
        labels = ['Uniform OK', 'Violation']
        sizes = [allowed, violations]
        colors = [COLOR_SUCCESS, COLOR_DANGER]
        
        if total == 0:
            ax.text(0.5, 0.5, "No Data Available", ha='center', va='center', color='white', fontsize=14)
            ax.axis('off')
        else:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
            for text in texts + autotexts: text.set_color('white')
        ax.axis('equal')

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        plt.close(fig)

        # Update Tree
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for index, row in self.data.iterrows():
            tag = 'allowed' if row['Status'] == 'Allowed' else 'violation'
            self.tree.insert("", 0, values=(row['Date'], row['Time'], row['Status'], row['Confidence'], row['Notes']), tags=(tag,))
        
        self.tree.tag_configure('allowed', foreground=COLOR_SUCCESS)
        self.tree.tag_configure('violation', foreground=COLOR_DANGER)

    def on_close(self):
        self.window.destroy()

# =============================================================================
# UI MODULE: MAIN APPLICATION
# =============================================================================
class SafetyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Student Safety Monitor")
        self.root.geometry("1200x700")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(False, False)

        # Initialize Logic
        self.logger = DataLogger()
        self.detector = AIDetector()
        self.camera = cv2.VideoCapture(0)
        
        # Check camera
        if not self.camera.isOpened():
            messagebox.showerror("Camera Error", "Could not access the camera.")
        
        self.running = False
        self.violation_count = 0
        self.last_status = ""
        self.last_log_time = 0 # Debounce timer

        self.setup_ui()

    def setup_ui(self):
        # Header
        header_frame = Frame(self.root, bg=COLOR_BG, height=80)
        header_frame.pack(fill=X, side=TOP, padx=20, pady=10)
        
        Label(header_frame, text="AI Student Safety & Uniform Monitoring", 
              font=("Helvetica", 20, "bold"), bg=COLOR_BG, fg=COLOR_TEXT).pack(side=LEFT)
        
        self.status_indicator = Label(header_frame, text="● System Ready", 
                                 font=("Helvetica", 14), bg=COLOR_BG, fg=COLOR_ACCENT)
        self.status_indicator.pack(side=RIGHT)

        # Content
        content_frame = Frame(self.root, bg=COLOR_BG)
        content_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)

        # Camera Panel (Left)
        left_panel = Frame(content_frame, bg="#34495E", width=800, height=500)
        left_panel.pack(side=LEFT, fill=BOTH, expand=True)
        left_panel.pack_propagate(False)

        self.video_label = Label(left_panel, bg="black")
        self.video_label.pack(fill=BOTH, expand=True)

        # Dashboard Panel (Right)
        right_panel = Frame(content_frame, bg=COLOR_BG, width=350)
        right_panel.pack(side=RIGHT, fill=Y, padx=(20, 0))

        # Stats
        stats_frame = Frame(right_panel, bg="#34495E", padx=15, pady=15)
        stats_frame.pack(fill=X, pady=(0, 20))
        
        Label(stats_frame, text="VIOLATION COUNT", font=("Helvetica", 10), 
              bg="#34495E", fg="#BDC3C7").pack(anchor=W)
        self.violation_lbl = Label(stats_frame, text="0", font=("Helvetica", 32, "bold"), 
                                   bg="#34495E", fg=COLOR_DANGER)
        self.violation_lbl.pack(anchor=W)

        # Controls
        controls_frame = Frame(right_panel, bg=COLOR_BG)
        controls_frame.pack(fill=X, pady=(0, 20))

        self.btn_start = self.create_modern_button(controls_frame, "Start Camera", self.start_camera, COLOR_SUCCESS)
        self.btn_start.pack(fill=X, pady=5)

        self.btn_stop = self.create_modern_button(controls_frame, "Stop Camera", self.stop_camera, COLOR_DANGER)
        self.btn_stop.pack(fill=X, pady=5)
        self.btn_stop.config(state=DISABLED)

        self.btn_admin = self.create_modern_button(controls_frame, "Open Admin Panel", self.open_admin_panel, "#F39C12")
        self.btn_admin.pack(fill=X, pady=5)

        self.btn_logs = self.create_modern_button(controls_frame, "View Logs CSV", self.view_logs, COLOR_ACCENT)
        self.btn_logs.pack(fill=X, pady=5)

        # Activity List
        Label(right_panel, text="Recent Activity", font=("Helvetica", 14, "bold"), 
              bg=COLOR_BG, fg=COLOR_TEXT).pack(anchor=W)
        
        columns = ("time", "status")
        self.tree = ttk.Treeview(right_panel, columns=columns, show="headings", height=12)
        self.tree.heading("time", text="Time")
        self.tree.heading("status", text="Status")
        self.tree.column("time", width=80)
        self.tree.column("status", width=100)
        
        scrollbar = ttk.Scrollbar(right_panel, orient=VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        self.tree.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

    def create_modern_button(self, parent, text, command, color):
        return Button(parent, text=text, command=command, bg=color, fg="white",
                     font=("Helvetica", 12, "bold"), relief=FLAT, cursor="hand2", padx=10, pady=8)

    def start_camera(self):
        if not self.running:
            self.running = True
            self.btn_start.config(state=DISABLED)
            self.btn_stop.config(state=NORMAL)
            self.status_indicator.config(text="● Active Monitoring", fg=COLOR_SUCCESS)
            # Use root.after for thread-safe UI updates
            self.update_frame()

    def stop_camera(self):
        self.running = False
        self.btn_start.config(state=NORMAL)
        self.btn_stop.config(state=DISABLED)
        self.status_indicator.config(text="● Paused", fg=COLOR_WARN)
        self.video_label.config(image='', bg="black") # Reset background

    def update_frame(self):
        """ Thread-safe video loop using Tkinter's scheduler """
        if self.running:
            ret, frame = self.camera.read()
            if ret:
                frame, status, conf = self.detector.process_frame(frame)

                # Logging (Debounced: log only every 3 seconds max to avoid spam)
                now_ts = datetime.now().timestamp()
                if status in ["Allowed", "Violation"]:
                    if status != self.last_status or (now_ts - self.last_log_time > 3.0):
                        notes = "Uniform OK" if status == "Allowed" else "Dress Code Violation"
                        hour = datetime.now().hour
                        if hour >= 9 and status == "Allowed": notes += " (Late Entry)"
                        
                        self.logger.log_entry(status, conf, notes)
                        self.update_tree(status, datetime.now().strftime("%H:%M:%S"))
                        
                        if status == "Violation":
                            self.violation_count += 1
                            self.violation_lbl.config(text=str(self.violation_count))
                        
                        self.last_status = status
                        self.last_log_time = now_ts

                # Visual Feedback: Flash background red on violation
                bg_color = "#2C0000" if status == "Violation" else "black"
                self.video_label.config(bg=bg_color)

                # Resize and Convert
                frame = cv2.resize(frame, (800, 500))
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                
                self.video_label.img_tk = img_tk
                self.video_label.configure(image=img_tk)

            # Schedule next frame (30ms ~ 33FPS)
            self.root.after(30, self.update_frame)

    def update_tree(self, status, time_str):
        tag = 'success' if status == "Allowed" else 'danger'
        self.tree.insert("", 0, values=(time_str, status), tags=(tag,))
        self.tree.tag_configure('success', foreground='green')
        self.tree.tag_configure('danger', foreground='red')

    def open_admin_panel(self):
        AdminPanel(self.root)

    def view_logs(self):
        try:
            os.startfile(LOG_FILE)
        except AttributeError:
            try: os.system(f'open {LOG_FILE}')
            except: os.system(f'xdg-open {LOG_FILE}')

    def on_closing(self):
        self.stop_camera()
        if self.camera.isOpened():
            self.camera.release()
        self.root.destroy()

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    root = Tk()
    app = SafetyApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()