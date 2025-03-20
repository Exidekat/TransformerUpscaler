import inspect
import sys
import threading
import time
import tkinter as tk
from collections import defaultdict

import pandas as pd

class TracebackWindow:
	def __init__(self, root):
		self.root = root
		self.root.title("Traceback Window")
		self.index_widget = tk.Text(root, wrap='word', height=1)
		self.index_widget.pack()
		self.text = tk.Text(root, wrap='word')
		self.text.pack(expand=True, fill='both')
		self.text.config(state=tk.DISABLED)
		self.text.focus_set()

		self.lock = False
		self.idx = -1
		self.log = pd.Series({0: "No data"})
		self.text.bind("<KeyRelease-space>", self.toggle_freeze)
		self.text.bind("<KeyRelease-Left>", lambda event: self.navigate(event, -1))
		self.text.bind("<KeyRelease-Right>", lambda event: self.navigate(event, +1))

		self.update_interval = 50  # Update every 50 ms (0.05 second)
		self.function_index_widgets = {}  # Dictionary to store function call index_widgets
		self.function_last_active = {}
		self.function_net_time = {}
		self.update_traceback()  # this function must always be the last

	def toggle_freeze(self, event):
		self.lock = not self.lock
		self.idx = -1 if not self.lock else self.idx
		self.update_traceback()

	def navigate(self, event, factor):
		self.idx = min(max(self.idx + factor, 0), self.log.size - 1)
		self.update_traceback()

	def update_traceback(self):
		stack = self.get_current_stack()
		current_time = time.time()
		stack_trace = []

		for frame in stack:
			filename, lineno, function, code, depth = frame

			# Create a unique key for the current function call
			frame_key = (filename, lineno, function, depth)

			# Check if the function call is already in the dictionary
			if depth in self.function_last_active:
				if frame_key != self.function_last_active[depth]:
					self.function_index_widgets[frame_key] = current_time
					ended_frame = self.function_last_active[depth]
					self.function_last_active[depth] = frame_key

					if ended_frame not in self.function_net_time:
						self.function_net_time[ended_frame] = current_time - self.function_index_widgets[ended_frame]
					else:
						self.function_net_time[ended_frame] += current_time - self.function_index_widgets[ended_frame]
			else:
				self.function_last_active[depth] = frame_key
				self.function_index_widgets[frame_key] = current_time

			# Format stack string
			time_spent = current_time - self.function_index_widgets[frame_key]
			SHOW_ONLY_REPEATS = True
			if frame_key in self.function_net_time:
				#net_time_spent_on_function = self.function_net_time[frame_key] + time_spent
				#frame_trace = f"net {net_time_spent_on_function:.2f} : current {time_spent:.2f} seconds: {filename}, line {lineno}, in {function}\n	{code}\n"
				#stack_trace.append(("red", frame_trace))
				frame_trace = f"{time_spent:.2f} seconds: {filename}, line {lineno}, in {function}\n	{code}\n"
				stack_trace.append((None, frame_trace))
			elif not SHOW_ONLY_REPEATS:
				frame_trace = f"{time_spent:.2f} seconds: {filename}, line {lineno}, in {function}\n	{code}\n"
				stack_trace.append((None, frame_trace))
		self.log[self.log.size] = "\n".join([(tag + trace if tag is not None else trace) for tag, trace in stack_trace])

		self.index_widget.config(state=tk.NORMAL)
		self.index_widget.delete('1.0', tk.END)
		self.index_widget.insert(tk.END, f"{self.idx} [{self.lock}]")
		self.index_widget.config(state=tk.DISABLED)

		if self.lock:
			self.root.after(self.update_interval, self.update_traceback)
			return

		self.text.config(state=tk.NORMAL)
		self.text.delete('1.0', tk.END)
		self.text.insert(tk.END, self.log.iloc[self.idx])
		self.text.config(state=tk.DISABLED)

		if project_thread and not project_thread.is_alive():
			self.print_ordered_index_widgets()
			self.root.quit()
		else:
			self.root.after(self.update_interval, self.update_traceback)

	def get_current_stack(self):
		if project_thread is None or not project_thread.is_alive():
			return []

		frames = []
		current_frame = sys._current_frames().get(project_thread.ident)
		while current_frame:
			frames.append(current_frame)
			current_frame = current_frame.f_back
		frames.reverse()  # From the outermost call to innermost call

		stack_summary = []
		for depth, frame in enumerate(frames):
			frame_info = inspect.getframeinfo(frame)
			# Extract the required four elements for traceback.format_list
			stack_summary.append((frame_info.filename, frame_info.lineno, frame_info.function,
								  frame_info.code_context[0].strip() if frame_info.code_context else '', depth))

		return stack_summary

	def print_ordered_index_widgets(self):
		# Group function calls by depth
		depth_groups = defaultdict(list)
		for frame_key, index_widget in self.function_index_widgets.items():
			filename, lineno, function, depth = frame_key
			time_spent = time.time() - index_widget
			depth_groups[depth].append((time_spent, filename, lineno, function))

		# Sort each depth group by time spent (greatest to least)
		for depth in depth_groups:
			depth_groups[depth].sort(reverse=True, key=lambda x: x[0])

		# Print the results
		PRINT_TRACEBACK_TIMES = True
		if PRINT_TRACEBACK_TIMES:
			print("Function call times ordered by greatest time spent:")
			for depth in sorted(depth_groups.keys()):
				print(f"\nStack depth {depth}:")
				for time_spent, filename, lineno, function in depth_groups[depth]:
					print(f"{time_spent:.2f} seconds: {filename}, line {lineno}, in {function}")


project_thread = None


def traceback_display(func):
	def wrapper(*args, **kwargs):
		global project_thread
		root = tk.Tk()
		traceback_window = TracebackWindow(root)

		# Run start_project in a separate thread
		project_thread = threading.Thread(target=func)
		project_thread.start()

		root.mainloop()

	return wrapper
