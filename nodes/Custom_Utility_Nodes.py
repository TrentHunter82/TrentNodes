# A minimal, self-contained script providing a Number Counter and a Text File Line Loader.
# Based on code originally by Jordan Thompson (WASasquatch) from the WAS Node Suite.
#
# Original Copyright 2023 Jordan Thompson (WASasquatch)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to
# deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import json
import hashlib

MANIFEST = {
    "name": "Custom Utility Nodes",
    "version": (1, 0, 1),
    "author": "Customized by User",
    "project": "#",
    "description": "A custom implementation of the Number Counter and Text Load Line From File nodes.",
}

# HELPER CLASS FOR COLORED CONSOLE OUTPUT
class CUN_cstr:
    class color:
        END = '\33[0m'
        BLUE = '\33[34m'
        YELLOW = '\33[33m'
        RED = '\33[31m'
    
    @staticmethod
    def print(message, level="info"):
        color_code = ""
        prefix = "Custom Node: "
        if level == "warning":
            color_code = CUN_cstr.color.YELLOW
            prefix = "Custom Node Warning: "
        elif level == "error":
            color_code = CUN_cstr.color.RED
            prefix = "Custom Node Error: "
        else:
            color_code = CUN_cstr.color.BLUE

        print(f"{color_code}{prefix}{CUN_cstr.color.END}{message}")

# GLOBALS
NODE_FILE = os.path.abspath(__file__)
CUSTOM_NODE_ROOT = os.path.dirname(NODE_FILE)
CUSTOM_NODE_DATABASE = os.path.join(CUSTOM_NODE_ROOT, 'custom_utility_nodes_database.json')
TEXT_TYPE = "STRING"

# A SIMPLE KEY-VALUE DATABASE FOR STORING NODE STATE
class CUN_SimpleJSONDB:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, IOError):
                CUN_cstr.print(f"Database file '{filepath}' is corrupted or unreadable. Starting fresh.", "warning")
                self.data = {}
        else:
            self.data = {}
            
    def get(self, category, key):
        return self.data.get(category, {}).get(key, None)

    def insert(self, category, key, value):
        if category not in self.data:
            self.data[category] = {}
        self.data[category][key] = value
        self._save()

    def _save(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.data, f, indent=4)
        except Exception as e:
            CUN_cstr.print(f"Error saving database file: {e}", "error")

DB = CUN_SimpleJSONDB(CUSTOM_NODE_DATABASE)

# NODE IMPLEMENTATIONS

class CUN_NumberCounter:
    def __init__(self):
        self.counters = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number_type": (["integer", "float"],),
                "mode": (["increment", "decrement", "increment_to_stop", "decrement_to_stop", "reset_after_stop"],),
                "start": ("FLOAT", {"default": 0, "min": -1e18, "max": 1e18, "step": 0.01}),
                "stop": ("FLOAT", {"default": 100, "min": -1e18, "max": 1e18, "step": 0.01}),
                "step": ("FLOAT", {"default": 1, "min": 0, "max": 1e6, "step": 0.01}),
            },
            "optional": {
                "reset_bool": ("NUMBER", {"default": 0}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("NUMBER", "FLOAT", "INT")
    RETURN_NAMES = ("number", "float", "int")
    FUNCTION = "increment_number"
    CATEGORY = "Trent Nodes"

    def increment_number(self, number_type, mode, start, stop, step, unique_id, reset_bool=0):
        counter = float(start)
        if unique_id in self.counters:
            counter = self.counters[unique_id]

        if round(reset_bool) >= 1:
            counter = start

        if mode == 'increment':
            counter += step
        elif mode == 'decrement':
            counter -= step
        elif mode == 'increment_to_stop':
            if counter < stop:
                counter += step
        elif mode == 'decrement_to_stop':
            if counter > stop:
                counter -= step
        elif mode == 'reset_after_stop':
            if counter < stop:
                counter += step
            else:
                counter = start
        
        self.counters[unique_id] = counter
        result = int(counter) if number_type == 'integer' else float(counter)
        return (result, float(counter), int(counter))

class CUN_TextFileLineLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": '', "multiline": False}),
                "label": ("STRING", {"default": 'TextBatch', "multiline": False}),
                "mode": (["automatic", "index"],),
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, file_path, mode, **kwargs):
        if mode != 'index':
            return float("NaN")
        try:
            m = hashlib.sha256()
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    m.update(f.read())
                return m.digest().hex()
        except Exception:
            return float("NaN")
        return float("NaN")

    RETURN_TYPES = (TEXT_TYPE, "INT")
    RETURN_NAMES = ("line_text", "line_count")
    FUNCTION = "load_file_line"
    CATEGORY = "Custom_Utility_Nodes"

    def load_file_line(self, file_path='', label='TextBatch', mode='automatic', index=0):
        if not file_path or not os.path.exists(file_path):
            CUN_cstr.print(f"File not found at path: {file_path}", "error")
            return ('', 0)

        file_loader = self.TextFileLoader(file_path, label)
        
        line_count = len(file_loader.lines)
        if line_count == 0:
            CUN_cstr.print("File is empty.", "warning")
            return ('', 0)
        
        line = ''
        if mode == 'automatic':
            line = file_loader.get_next_line()
        elif mode == 'index':
            line = file_loader.get_line_by_index(index)
        
        if line is None:
            return ('', line_count)

        return (line, line_count)

    class TextFileLoader:
        def __init__(self, file_path, label):
            self.file_path = file_path
            self.label = label
            self.lines = []
            self.index = 0
            self._load_state()

        def _load_state(self):
            stored_file_path = DB.get('CUN_TextBatch_Paths', self.label)
            stored_index = DB.get('CUN_TextBatch_Counters', self.label)
            
            if stored_file_path != self.file_path:
                self.index = 0
            else:
                self.index = stored_index if stored_index is not None else 0
            
            try:
                with open(self.file_path, 'r', encoding="utf-8", newline='\n') as file:
                    self.lines = [line.strip() for line in file if line.strip()]
            except Exception as e:
                CUN_cstr.print(f"Could not read file: {e}", "error")

        def get_next_line(self):
            if not self.lines: return None
            current_index = self.index
            if current_index >= len(self.lines):
                current_index = 0
            
            line = self.lines[current_index]
            self.index = (current_index + 1) % len(self.lines)
            self._save_state()
            return line

        def get_line_by_index(self, idx):
            if not self.lines:
                CUN_cstr.print("Cannot get line by index from an empty file.", "warning")
                return None
            
            safe_idx = idx % len(self.lines)
            if not (0 <= safe_idx < len(self.lines)):
                CUN_cstr.print(f"Index {safe_idx} is out of bounds for file with {len(self.lines)} lines.", "error")
                return None

            self.index = safe_idx
            self._save_state()
            return self.lines[safe_idx]

        def _save_state(self):
            DB.insert('CUN_TextBatch_Paths', self.label, self.file_path)
            DB.insert('CUN_TextBatch_Counters', self.label, self.index)

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "CUN_NumberCounter": CUN_NumberCounter,
    "CUN_TextFileLineLoader": CUN_TextFileLineLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CUN_NumberCounter": "Number Counter",
    "CUN_TextFileLineLoader": "Text File Line Loader",
}