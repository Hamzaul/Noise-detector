"""
Noise Control Detector – Tkinter GUI (AI & DS Hackathon)
--------------------------------------------------------
• Live microphone monitor with color status (Silent / Warning / Too Noisy)
• Adjustable thresholds + one-click ambient calibration
• Device picker (choose correct mic)
• Realtime chart of dBFS levels

Dependencies:
  pip install sounddevice numpy matplotlib
Run:
  python noise_gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
from collections import deque
import os
import numpy as np
import sounddevice as sd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import winsound   # For beep sound on Windows


class NoiseDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Noise Control Detector – Library/Classroom")
        self.root.geometry("900x600")
        self.beeping = False
        self.beeped = False  # track if we already beeped for current noise
        self.calibrating = False

        # Audio config
        self.fs = 44100            # Sample rate
        self.window_sec = 0.2      # Analysis window in seconds (200 ms)
        self.blocksize = int(self.fs * self.window_sec)
        self.stream = None
        self.audio_q = queue.Queue()
        self.running = False

        # History for plotting (store ~30s at 10 Hz)
        self.history_len = 300
        self.db_history = deque(maxlen=self.history_len)

        # Thresholds (in dBFS; -90..0). Less negative = louder.
        self.silent_threshold = tk.DoubleVar(value=-20.0)
        self.noisy_threshold = tk.DoubleVar(value=-10.0)

        # UI build
        self._build_ui()

        # Populate devices and set defaults
        self._refresh_devices()

        # Periodic UI update
        self._schedule_ui_update()

    # ---------------------------- Beep Helper ----------------------------
    def play_beep(self):
        """Play a beep sound when noise threshold is exceeded."""
        if self.beeping:
            return
        self.beeping = True

        def _beep():
            try:
                beep_path = os.path.join(os.path.dirname(__file__), "beep.wav")
                if os.path.exists(beep_path):
                    winsound.PlaySound(beep_path, winsound.SND_FILENAME)
                else:
                    # Fallback tone if beep.wav not found
                    winsound.Beep(1000, 300)
            finally:
                self.beeping = False

        threading.Thread(target=_beep, daemon=True).start()

    # ---------------------------- UI ----------------------------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill=tk.X)

        title = ttk.Label(top, text="📢 Noise Control Detector", font=("Segoe UI", 18, "bold"))
        title.pack(side=tk.LEFT)

        about_btn = ttk.Button(top, text="About", command=self._show_about)
        about_btn.pack(side=tk.RIGHT)

        # Controls frame
        controls = ttk.LabelFrame(self.root, text="Controls", padding=12)
        controls.pack(fill=tk.X, padx=12, pady=8)

        # Device picker
        ttk.Label(controls, text="Input Device:").grid(row=0, column=0, sticky=tk.W, padx=(0,6), pady=4)
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(controls, textvariable=self.device_var, state="readonly", width=40)
        self.device_combo.grid(row=0, column=1, sticky=tk.W, pady=4)
        ttk.Button(controls, text="Refresh", command=self._refresh_devices).grid(row=0, column=2, padx=6)

        # Start/Stop
        self.start_btn = ttk.Button(controls, text="Start", command=self.start)
        self.stop_btn = ttk.Button(controls, text="Stop", command=self.stop, state=tk.DISABLED)
        self.start_btn.grid(row=0, column=3, padx=(12,6))
        self.stop_btn.grid(row=0, column=4)

        # Automatic calibration button
        ttk.Button(controls, text="Calibrate", command=self.calibrate_thresholds).grid(row=0, column=5, padx=6)

        # Threshold sliders
        sliders = ttk.Frame(controls)
        sliders.grid(row=1, column=0, columnspan=6, sticky=tk.W, pady=(8,0))
        ttk.Label(sliders, text="Warning threshold (dBFS)").grid(row=0, column=0, sticky=tk.W)
        self.silent_scale = ttk.Scale(sliders, from_=-80, to=0, orient=tk.HORIZONTAL,
                                      variable=self.silent_threshold, length=300,
                                      command=lambda v: self._sync_thresholds(order="silent"))
        self.silent_scale.grid(row=0, column=1, padx=8)
        self.silent_val_lbl = ttk.Label(sliders, text=f"{self.silent_threshold.get():.1f}")
        self.silent_val_lbl.grid(row=0, column=2, padx=(0,20))

        ttk.Label(sliders, text="Too-Noisy threshold (dBFS)").grid(row=0, column=3, sticky=tk.W)
        self.noisy_scale = ttk.Scale(sliders, from_=-80, to=0, orient=tk.HORIZONTAL,
                                     variable=self.noisy_threshold, length=300,
                                     command=lambda v: self._sync_thresholds(order="noisy"))
        self.noisy_scale.grid(row=0, column=4, padx=8)
        self.noisy_val_lbl = ttk.Label(sliders, text=f"{self.noisy_threshold.get():.1f}")
        self.noisy_val_lbl.grid(row=0, column=5)

        # Status frame
        status_frame = ttk.LabelFrame(self.root, text="Live Status", padding=12)
        status_frame.pack(fill=tk.X, padx=12, pady=8)

        # Big status label
        self.status_lbl = tk.Label(status_frame, text="Idle", font=("Segoe UI", 24, "bold"), fg="gray")
        self.status_lbl.pack(anchor=tk.CENTER, pady=4)

        # Current dB label
        self.db_lbl = ttk.Label(status_frame, text="dBFS: —", font=("Segoe UI", 12))
        self.db_lbl.pack(anchor=tk.CENTER, pady=(0,8))

        # Progress bar (maps -80..0 dBFS to 0..100)
        self.pb = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, length=600,
                                  mode='determinate', maximum=100)
        self.pb.pack(anchor=tk.CENTER, pady=8)

        # Chart frame
        chart_frame = ttk.LabelFrame(self.root, text="Realtime Level (last ~30s)", padding=12)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0,12))

        self.fig = Figure(figsize=(8.5, 3.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim(-80, 0)
        self.ax.set_xlim(0, self.history_len)
        self.ax.set_ylabel("dBFS")
        self.ax.set_xlabel("Samples (≈0.1s)")
        self.line, = self.ax.plot([], [], lw=1.5)
        self.warn_line = self.ax.axhline(self.silent_threshold.get(), linestyle='--', color='orange', label='Warning')
        self.noisy_line = self.ax.axhline(self.noisy_threshold.get(), linestyle='--', color='red', label='Too-Noisy')
        self.ax.legend(loc='lower right')

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _show_about(self):
        message = (
            "Noise Control Detector (Tkinter)\n\n"
            "• Uses microphone input to estimate loudness in dBFS (relative to full scale).\n"
            "• Adjust thresholds to adapt to the room.\n"
            "• Green = Silent, Orange = Warning, Red = Too Noisy.\n\n"
            "Tip: Pick the correct mic device and allow OS permissions."
        )
        messagebox.showinfo("About", message)

    def _sync_thresholds(self, order="silent"):
        """Keep thresholds consistent and update labels/lines."""
        s = self.silent_threshold.get()
        n = self.noisy_threshold.get()
        if order == "silent" and s > n - 1.0:
            self.noisy_threshold.set(s + 1.0)
        elif order == "noisy" and n < s + 1.0:
            self.silent_threshold.set(n - 1.0)

        self.silent_val_lbl.config(text=f"{self.silent_threshold.get():.1f}")
        self.noisy_val_lbl.config(text=f"{self.noisy_threshold.get():.1f}")
        self.warn_line.set_ydata([self.silent_threshold.get(), self.silent_threshold.get()])
        self.noisy_line.set_ydata([self.noisy_threshold.get(), self.noisy_threshold.get()])
        self.canvas.draw_idle()

    # ---------------------------- Audio ----------------------------
    def _refresh_devices(self):
        try:
            devices = sd.query_devices()
        except Exception as e:
            messagebox.showerror("Audio Error", f"Could not query audio devices:\n{e}")
            return

        input_devices = []
        for idx, d in enumerate(devices):
            if d.get('max_input_channels', 0) > 0:
                name = d.get('name', f'Device {idx}')
                label = f"[{idx}] {name}"
                input_devices.append(label)

        if not input_devices:
            messagebox.showwarning("No Microphone",
                                   "No input devices found. Connect/enable a microphone and click Refresh.")
        self.device_combo['values'] = input_devices
        if input_devices:
            self.device_combo.set(input_devices[0])

    def start(self):
        if self.running:
            return
        device_label = self.device_var.get()
        if not device_label:
            messagebox.showwarning("Select Device", "Please select an input device first.")
            return
        try:
            device_idx = int(device_label.split(']')[0].strip('['))
        except Exception:
            device_idx = None

        try:
            self.stream = sd.InputStream(
                samplerate=self.fs,
                channels=1,
                dtype='float32',
                blocksize=self.blocksize,
                device=device_idx,
                callback=self._audio_callback,
            )
            self.stream.start()
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_lbl.config(text="Listening…", fg="#1f77b4")
        except Exception as e:
            messagebox.showerror("Audio Error", f"Could not start stream:\n{e}")

    def stop(self):
        if not self.running:
            return
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        finally:
            self.stream = None
            self.running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_lbl.config(text="Stopped", fg="gray")

    def _audio_callback(self, indata, frames, time_info, status):
        rms = float(np.sqrt(np.mean(np.square(indata))))
        dbfs = 20.0 * np.log10(rms + 1e-12)
        try:
            self.audio_q.put_nowait(dbfs)
        except queue.Full:
            pass

    # ---------------------------- Update Loop ----------------------------
    def _schedule_ui_update(self):
        self.root.after(80, self._update_ui)  # ~12.5 Hz

    def _update_ui(self):
        last = None
        try:
            while True:
                last = self.audio_q.get_nowait()
        except queue.Empty:
            pass

        if last is not None:
            self.db_history.append(last)
            self.db_lbl.config(text=f"dBFS: {last:.1f}")
            level = int(np.clip((last + 80.0) / 80.0 * 100.0, 0, 100))
            self.pb['value'] = level

            # Status + beep
            if last >= self.noisy_threshold.get():  # <-- fixed threshold trigger
                self.status_lbl.config(text="❌ Stop Noise (BEEP!)", fg="red")
                if not self.beeped:
                    self.play_beep()
                    self.beeped = True
            else:
                self.beeped = False
                if last >= self.silent_threshold.get():
                    self.status_lbl.config(text="⚠️ Warning", fg="orange")
                else:
                    self.status_lbl.config(text="✅ Silent", fg="green")

            # Update chart
            y = list(self.db_history)
            x = list(range(len(y)))
            self.line.set_data(x, y)
            self.ax.set_xlim(max(0, len(y) - self.history_len), max(self.history_len, len(y)))
            self.warn_line.set_ydata([self.silent_threshold.get(), self.silent_threshold.get()])
            self.noisy_line.set_ydata([self.noisy_threshold.get(), self.noisy_threshold.get()])
            self.canvas.draw_idle()

        self._schedule_ui_update()

    # ---------------------------- Automatic Calibration ----------------------------
    def calibrate_thresholds(self, duration=2.0):
        """Automatically set thresholds based on ambient noise."""
        if self.running:
            messagebox.showinfo("Calibration", "Stop current session before calibrating.")
            return

        messagebox.showinfo("Calibration", "Measuring ambient noise for a few seconds...")
        try:
            device_label = self.device_var.get()
            device_idx = int(device_label.split(']')[0].strip('[')) if device_label else None

            recording = sd.rec(int(self.fs*duration), samplerate=self.fs, channels=1, dtype='float32', device=device_idx)
            sd.wait()
            rms = np.sqrt(np.mean(np.square(recording)))
            ambient_db = 20*np.log10(rms + 1e-12)

            # Set thresholds based on ambient
            self.silent_threshold.set(ambient_db + 5)  # Warning threshold slightly above ambient
            self.noisy_threshold.set(ambient_db + 15)  # Too noisy threshold higher
            self._sync_thresholds(order="silent")
            messagebox.showinfo("Calibration Done", f"Ambient noise: {ambient_db:.1f} dBFS\nThresholds set automatically.")
        except Exception as e:
            messagebox.showerror("Calibration Error", f"Could not calibrate:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.25)
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    app = NoiseDetectorApp(root)
    root.mainloop()
