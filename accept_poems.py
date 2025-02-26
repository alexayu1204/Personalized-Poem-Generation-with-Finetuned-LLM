import json

with open("accept_poems.json", "r", encoding="utf-8") as f:
    results = json.load(f)

import tkinter as tk
from tkinter import ttk, messagebox
import json

# Sample 'results' data for demonstration
#results = [
#    [
#        ['叶子', '叶子，叶子\n叶子，叶子'],
#        ['河水', '在夜的深处\n河水\n是黑暗的\n眼睛'],
#        ['积雪', '在落雪的夜晚\n我将积雪\n藏在了心里'],
#        ['虫子', '在夜的深处\n虫子\n用尽全身的力气\n嘶喊\n直到\n沙哑'],
#        ['山峰', '在山峰的顶端\n我将自己\n压成一片叶子']
#    ],
#    [
#        ['冷风', '在风的边缘\n我学会了\n如何颤抖'],
#        ['雨水中的伞', '在雨的细语中\n我撑开伞\n遮挡了\n从天而降的思念'],
#        ['破碎的树枝', '在风的缝隙中\n我看见\n破碎的树枝'],
#        ['轻舞的枝条', '轻舞的枝条\n在风中\n寻找着\n属于自己的\n一条路'],
#        ['碎裂的玻璃', '那些碎裂的玻璃\n在夜晚的风中\n闪烁着\n绝望的光芒'],
#        ['风中的落叶', '风中的落叶，\n是岁月的叹息，\n在空中盘旋，\n寻找归宿的勇气。']
#    ],
#    [
#        ['枯萎的莲花', '在枯萎的莲花里\n藏着一个\n干涸的\n海'],
#        ['孤雁的轨迹', '在无尽的夜空里\n我寻找着\n那一抹\n孤雁的轨迹']
#    ]
#    # ... more inner lists
#]

class EditDialog(tk.Toplevel):
    def __init__(self, master, poem, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.title("Edit Poem")
        self.geometry("400x300")
        self.poem = poem
        self.result = None  # To store the edited data

        # Make the dialog modal
        self.transient(master)
        self.grab_set()

        # Variables to track what to edit
        self.edit_title_var = tk.BooleanVar()
        self.edit_body_var = tk.BooleanVar()

        # Checkboxes to choose what to edit
        self.checkbox_frame = ttk.Frame(self)
        self.checkbox_frame.pack(pady=10, padx=10, fill='x')

        self.title_checkbox = ttk.Checkbutton(self.checkbox_frame, text="Edit Title", variable=self.edit_title_var, command=self.toggle_fields)
        self.title_checkbox.pack(anchor='w')

        self.body_checkbox = ttk.Checkbutton(self.checkbox_frame, text="Edit Body", variable=self.edit_body_var, command=self.toggle_fields)
        self.body_checkbox.pack(anchor='w')

        # Entry fields
        self.fields_frame = ttk.Frame(self)
        self.fields_frame.pack(pady=10, padx=10, fill='both', expand=True)

        # Title field
        self.title_label = ttk.Label(self.fields_frame, text="Title:")
        self.title_label.grid(row=0, column=0, sticky='nw')
        self.title_entry = ttk.Entry(self.fields_frame, width=50)
        self.title_entry.grid(row=0, column=1, pady=5, padx=5, sticky='nw')

        # Body field
        self.body_label = ttk.Label(self.fields_frame, text="Body:")
        self.body_label.grid(row=1, column=0, sticky='nw')
        self.body_text = tk.Text(self.fields_frame, width=50, height=10, wrap='word')
        self.body_text.grid(row=1, column=1, pady=5, padx=5, sticky='nw')

        # Initially disable fields
        self.title_entry.config(state='disabled')
        self.body_text.config(state='disabled')

        # Buttons
        self.buttons_frame = ttk.Frame(self)
        self.buttons_frame.pack(pady=10)

        self.save_button = ttk.Button(self.buttons_frame, text="Save", command=self.on_save)
        self.save_button.pack(side='left', padx=5)

        self.cancel_button = ttk.Button(self.buttons_frame, text="Cancel", command=self.on_cancel)
        self.cancel_button.pack(side='left', padx=5)

    def toggle_fields(self):
        if self.edit_title_var.get():
            self.title_entry.config(state='normal')
            self.title_entry.delete(0, tk.END)
            self.title_entry.insert(0, self.poem['image'])
        else:
            self.title_entry.config(state='disabled')

        if self.edit_body_var.get():
            self.body_text.config(state='normal')
            self.body_text.delete(1.0, tk.END)
            self.body_text.insert(tk.END, self.poem['response'])
        else:
            self.body_text.config(state='disabled')

    def on_save(self):
        new_title = self.poem['image']
        new_response = self.poem['response']

        if self.edit_title_var.get():
            edited_title = self.title_entry.get().strip()
            if edited_title:
                new_title = edited_title
            else:
                messagebox.showwarning("Input Error", "Title cannot be empty.")
                return

        if self.edit_body_var.get():
            edited_body = self.body_text.get(1.0, tk.END).strip()
            if edited_body:
                new_response = edited_body
            else:
                messagebox.showwarning("Input Error", "Body cannot be empty.")
                return

        self.result = {'image': new_title, 'response': new_response}
        self.destroy()

    def on_cancel(self):
        self.destroy()

class PoemReviewerApp:
    def __init__(self, master, results):
        self.master = master
        self.master.title("Poem Reviewer")
        self.master.geometry("700x500")
        self.master.resizable(False, False)

        # Initialize data
        self.flattened_results = results
        self.current_index = 0
        self.total = len(self.flattened_results)

        # GUI Components
        self.create_widgets()
        self.display_current_poem()

    def flatten_results(self, results):
        flattened = []
        for outer_idx, inner_list in enumerate(results):
            for inner_idx, pair in enumerate(inner_list):
                flattened.append({
                    'outer_idx': outer_idx,
                    'inner_idx': inner_idx,
                    'image': pair[0],
                    'response': pair[1],
                    'accepted': False,
                    'edited': False
                })
        return flattened

    def create_widgets(self):
        # Progress Frame
        self.progress_frame = ttk.Frame(self.master)
        self.progress_frame.pack(pady=10, padx=10, fill='x')

        self.progress_label = ttk.Label(self.progress_frame, text=f"Poem 0 of {self.total}")
        self.progress_label.pack(anchor='w')

        self.progress_bar = ttk.Progressbar(self.progress_frame, length=600, mode='determinate', maximum=self.total)
        self.progress_bar.pack(pady=5, fill='x')

        # Title Label
        self.title_label = ttk.Label(self.master, text="", font=("Helvetica", 16, "bold"), wraplength=680, justify='center')
        self.title_label.pack(pady=10)

        # Body Text
        self.body_text = tk.Text(self.master, wrap='word', height=15, width=80, state='disabled', font=("Helvetica", 12))
        self.body_text.pack(pady=10, padx=10)

        # Buttons Frame
        self.buttons_frame = ttk.Frame(self.master)
        self.buttons_frame.pack(pady=10)

        # Accept Button
        self.accept_button = ttk.Button(self.buttons_frame, text="Accept", command=self.accept_poem)
        self.accept_button.grid(row=0, column=0, padx=10, ipadx=10)

        # Edit Button
        self.edit_button = ttk.Button(self.buttons_frame, text="Edit", command=self.edit_poem)
        self.edit_button.grid(row=0, column=1, padx=10, ipadx=10)

        # Skip Button
        self.skip_button = ttk.Button(self.buttons_frame, text="Skip", command=self.skip_poem)
        self.skip_button.grid(row=0, column=2, padx=10, ipadx=10)

        # Delete Button
        self.delete_button = ttk.Button(self.buttons_frame, text="Delete", command=self.delete_poem)
        self.delete_button.grid(row=0, column=3, padx=10, ipadx=10)

        # Save Button
        self.save_button = ttk.Button(self.master, text="Save All Changes", command=self.save_changes)
        self.save_button.pack(pady=10, ipadx=10)

    def display_current_poem(self):
        if self.current_index >= self.total:
            self.title_label.config(text="All poems have been reviewed.")
            self.body_text.config(state='normal')
            self.body_text.delete(1.0, tk.END)
            self.body_text.insert(tk.END, "You have reviewed all the poems.")
            self.body_text.config(state='disabled')
            self.disable_buttons()
            self.progress_label.config(text=f"Poem {self.total} of {self.total}")
            self.progress_bar['value'] = self.total
            return

        poem = self.flattened_results[self.current_index]
        self.progress_label.config(text=f"Poem {self.current_index + 1} of {self.total}")
        self.progress_bar['value'] = self.current_index

        self.title_label.config(text=poem['image'])
        self.body_text.config(state='normal')
        self.body_text.delete(1.0, tk.END)
        self.body_text.insert(tk.END, poem['response'])
        self.body_text.config(state='disabled')

    def disable_buttons(self):
        self.accept_button.state(['disabled'])
        self.edit_button.state(['disabled'])
        self.skip_button.state(['disabled'])
        self.delete_button.state(['disabled'])

    def accept_poem(self):
        poem = self.flattened_results[self.current_index]
        poem['accepted'] = True
        messagebox.showinfo("Accepted", f"Poem '{poem['image']}' has been accepted.")
        self.current_index += 1
        self.display_current_poem()

    def edit_poem(self):
        poem = self.flattened_results[self.current_index]
        edit_dialog = EditDialog(self.master, poem)
        self.master.wait_window(edit_dialog)

        if edit_dialog.result:
            poem['image'] = edit_dialog.result['image']
            poem['response'] = edit_dialog.result['response']
            poem['edited'] = True
            messagebox.showinfo("Edited", f"Poem '{poem['image']}' has been edited.")
            self.current_index += 1
            self.display_current_poem()
        else:
            # User cancelled the edit
            pass

    def skip_poem(self):
        poem = self.flattened_results[self.current_index]
        messagebox.showinfo("Skipped", f"Poem '{poem['image']}' has been skipped.")
        self.current_index += 1
        self.display_current_poem()

    def delete_poem(self):
        poem = self.flattened_results[self.current_index]
        confirm = messagebox.askyesno("Delete", f"Are you sure you want to delete poem '{poem['image']}'?")
        if confirm:
            del self.flattened_results[self.current_index]
            self.total -= 1
            messagebox.showinfo("Deleted", f"Poem '{poem['image']}' has been deleted.")
            # Adjust progress bar maximum
            self.progress_bar['maximum'] = self.total
            self.display_current_poem()

    def save_changes(self):
        if not self.flattened_results:
            messagebox.showwarning("No Data", "There are no poems to save.")
            return

        # Reconstruct the nested list based on original outer indices
        reconstructed = [[] for _ in range(self.total)]
        for poem in self.flattened_results:
            outer = poem['outer_idx']
            inner = poem['inner_idx']
            # Ensure that the inner list has enough elements
            while len(reconstructed[outer]) <= inner:
                reconstructed[outer].append(['', ''])
            reconstructed[outer][inner] = [poem['image'], poem['response']]

        # Save to JSON
        try:
            with open('updated_poems.json', 'w', encoding='utf-8') as f:
                json.dump(reconstructed, f, ensure_ascii=False, indent=4)
            messagebox.showinfo("Saved", "All changes have been saved to 'updated_poems.json'.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

def main():
    root = tk.Tk()
    app = PoemReviewerApp(root, results)
    root.mainloop()

if __name__ == "__main__":
    main()

