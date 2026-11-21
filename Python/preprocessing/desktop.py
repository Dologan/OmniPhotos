"""Desktop entry point for configuring and monitoring OmniPhotos jobs."""

from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import yaml


class Desktop:
    def __init__(self, root, config_file, input_path):
        self.root = root
        self.process = None
        self.log_lines = Queue(maxsize=512)
        self.config = tk.StringVar(value=config_file or "")
        self.input = tk.StringVar(value=input_path or "")
        self.viewer_exe = tk.StringVar()
        self.viewer_config = tk.StringVar()
        self.pc_vr = tk.BooleanVar()
        self.status = tk.StringVar(value="Choose a preprocessing configuration or a prepared dataset.")

        root.title("OmniPhotos")
        root.geometry("850x540")
        root.minsize(650, 400)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        process = ttk.LabelFrame(root, text="Preprocess", padding=10)
        process.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        process.columnconfigure(1, weight=1)
        self.path_field(process, 0, "Pipeline YAML", self.config, (("YAML", "*.yaml *.yml"),))
        self.new_project_button = ttk.Button(process, text="New project…", command=self.new_project)
        self.new_project_button.grid(row=0, column=3, padx=(6, 0), sticky="ew")
        self.path_field(process, 1, "Input override", self.input, (("Stitched video", "*.mp4 *.mov *.mkv"),))
        ttk.Button(process, text="Image folder…", command=self.choose_images).grid(row=1, column=3, padx=(6, 0))
        self.process_button = ttk.Button(process, text="Start processing", command=self.preprocess)
        self.process_button.grid(row=2, column=1, sticky="w", pady=(8, 0))

        viewer = ttk.LabelFrame(root, text="View prepared dataset", padding=10)
        viewer.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
        viewer.columnconfigure(1, weight=1)
        self.path_field(viewer, 0, "Viewer executable", self.viewer_exe)
        self.path_field(viewer, 1, "Dataset YAML", self.viewer_config, (("YAML", "*.yaml *.yml"),))
        ttk.Checkbutton(viewer, text="Use SteamVR (PC VR)", variable=self.pc_vr).grid(
            row=2, column=1, sticky="w", pady=(8, 0)
        )
        self.view_button = ttk.Button(viewer, text="Open dataset", command=self.view)
        self.view_button.grid(row=2, column=2, sticky="e", pady=(8, 0))

        self.log = scrolledtext.ScrolledText(root, wrap="word", state="disabled", height=10)
        self.log.grid(row=2, column=0, sticky="nsew", padx=12, pady=6)
        ttk.Label(root, textvariable=self.status).grid(row=3, column=0, sticky="w", padx=12, pady=(2, 12))

    def path_field(self, parent, row, label, variable, filetypes=()):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 8))
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=3, padx=(0, 6))

        def browse():
            path = filedialog.askopenfilename(parent=self.root, filetypes=(*filetypes, ("All files", "*")))
            if path:
                variable.set(path)

        ttk.Button(parent, text="Browse…", command=browse).grid(row=row, column=2, sticky="ew")

    def choose_images(self):
        path = filedialog.askdirectory(parent=self.root)
        if path:
            self.input.set(path)

    def new_project(self):
        folder = filedialog.askdirectory(parent=self.root, title="Choose a folder for this capture")
        if not folder:
            return
        project = Path(folder)
        if any((project / name).exists() for name in ("temp", "Capture", "Cache", "Config")):
            self.status.set("Choose a project folder without previous processing output.")
            return
        selected = self.input.get() or filedialog.askopenfilename(
            parent=self.root, title="Choose a stitched 360 video",
            filetypes=(("Stitched video", "*.mp4 *.mov *.mkv"), ("All files", "*")),
        )
        if not selected:
            return
        source = Path(selected).expanduser()
        if not source.exists() or source.suffix.lower() == ".insv":
            self.status.set("Choose a stitched video or image folder. Export INSV footage in Insta360 Studio first.")
            return
        executable = filedialog.askopenfilename(parent=self.root, title="Choose the native Preprocessing executable")
        if not executable:
            return
        template = Path(__file__).with_name("config_omniphotos.sample.yaml")
        config = yaml.safe_load(template.read_text(encoding="utf-8"))
        config["preprocessing.omniphotos.execution_file_path"] = executable
        config["preprocessing.input_path"] = str(source.resolve())
        config["preprocessing.input_type"] = "image" if source.is_dir() else "video"
        pipeline = project / "pipeline.yaml"
        try:
            with pipeline.open("x", encoding="utf-8") as file:
                yaml.safe_dump(config, file, sort_keys=False)
        except OSError as error:
            self.status.set(f"Could not create {pipeline.name}. {error}")
            return
        self.config.set(str(pipeline))
        self.input.set("")
        native = Path(executable)
        viewer = native.with_name("Viewer" + native.suffix)
        if viewer.is_file():
            self.viewer_exe.set(str(viewer))
        self.status.set(f"Project created for {source.name}. Start processing when ready.")

    def preprocess(self):
        config = Path(self.config.get()).expanduser()
        if not config.is_file():
            self.status.set("Choose an existing preprocessing YAML file.")
            return
        config = config.resolve()
        command = [sys.executable, "-u", str(Path(__file__).resolve().with_name("main.py")),
                   "--headless", "--config_file", str(config)]
        if self.input.get():
            input_path = Path(self.input.get()).expanduser()
            if not input_path.exists() or input_path.suffix.lower() == ".insv":
                self.status.set("Choose a stitched video or image folder. Export INSV footage in Insta360 Studio first.")
                return
            if any((config.parent / name).exists() for name in ("temp", "Capture", "Cache", "Config")):
                self.status.set("To change the input, use a configuration in a new project folder.")
                return
            command.extend(["--input", str(input_path.resolve())])
        self.start(command, config.parent, "Preprocessing", self.select_viewer_output)

    def select_viewer_output(self, directory):
        prepared = sorted((directory / "Config").glob("config-viewer-*.yaml"))
        if len(prepared) == 1:
            self.viewer_config.set(str(prepared[0]))

    def view(self):
        executable = Path(self.viewer_exe.get()).expanduser()
        dataset = Path(self.viewer_config.get()).expanduser()
        if not executable.is_file() or not dataset.is_file():
            self.status.set("Choose the Viewer executable and a prepared dataset YAML.")
            return
        command = [str(executable.resolve()), str(dataset.resolve())]
        if self.pc_vr.get():
            command.append("--vr")
        self.start(command, executable.resolve().parent, "Viewer")

    def start(self, command, directory, label, on_success=None):
        try:
            self.process = subprocess.Popen(
                command, cwd=directory, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            )
        except OSError as error:
            self.status.set(f"Could not start {label.lower()}. {error}")
            return
        self.process_button.state(["disabled"])
        self.new_project_button.state(["disabled"])
        self.view_button.state(["disabled"])
        self.status.set(f"{label} running…")
        Thread(target=self.read_log, args=(self.process.stdout,), daemon=True).start()
        self.poll_log(label, directory, on_success)

    def read_log(self, stream):
        try:
            with stream:
                for line in stream:
                    self.log_lines.put(line)
        finally:
            self.log_lines.put(None)

    def poll_log(self, label, directory, on_success):
        lines = []
        finished = False
        for _ in range(100):
            try:
                line = self.log_lines.get_nowait()
            except Empty:
                break
            if line is None:
                finished = True
                break
            lines.append(line)
        if lines:
            self.log.configure(state="normal")
            self.log.insert("end", "".join(lines))
            self.log.see("end")
            self.log.configure(state="disabled")
        if finished:
            code = self.process.wait()
            self.process = None
            self.process_button.state(["!disabled"])
            self.new_project_button.state(["!disabled"])
            self.view_button.state(["!disabled"])
            if code == 0 and on_success:
                on_success(directory)
            message = f"{label} finished." if code == 0 else f"{label} failed with exit code {code}. See the log."
            self.status.set(message)
        else:
            self.root.after(80, self.poll_log, label, directory, on_success)

    def close(self):
        if self.process is not None:
            messagebox.showinfo("OmniPhotos", "Wait for the current job to finish before closing.", parent=self.root)
        else:
            self.root.destroy()


def launch(config_file=None, input_path=None):
    root = tk.Tk()
    Desktop(root, config_file, input_path)
    root.mainloop()
