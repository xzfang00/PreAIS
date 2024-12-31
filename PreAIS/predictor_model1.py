import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
from model.CNN_DNN import model_DNN_CNN
from feature_extract.K_mer import kmer_feature
from Bio import SeqIO
import numpy as np


def accept(file_path):
    seq = []
    result = []
    for record in SeqIO.parse(file_path, "fasta"):
        seq.append(str(record.seq))
    data = kmer_feature(seq, 3)
    model = model_DNN_CNN(data.shape[1], 1)
    model.load_state_dict(torch.load('./ckpt_save/best.pth'))
    model.eval()
    data_tensor = torch.tensor(data, dtype=torch.float)
    probability = model(data_tensor)
    probability = probability.detach().numpy()
    probability = np.round(probability.flatten(), 2)

    for p in probability:
        if p >= 0.5:
            result.append("pos+")
        else:
            result.append("neg-")
    return seq, result, probability

def show_result(seq, result, probability):
    result_window = tk.Toplevel(root)
    result_window.title("Result")
    result_window.geometry("550x300")

    # 创建表格
    tree = ttk.Treeview(result_window, columns=('Seq', 'Result', 'Probability'), show='headings')
    tree.heading('Seq', text='Seq')
    tree.heading('Result', text='Result')
    tree.heading('Probability', text='Probability')

    # 设置列宽度
    tree.column('Seq', width=400, anchor='center')
    tree.column('Result', width=70, anchor='center')
    tree.column('Probability', width=70, anchor='center')

    for i in range(len(seq)):
        tree.insert('', 'end', values=(seq[i], result[i], probability[i]))

    # Scrollbar
    scrollbar = ttk.Scrollbar(result_window, orient='vertical', command=tree.yview)
    tree.configure(yscroll=scrollbar.set)

    # Pack tree and scrollbar
    tree.pack(expand=True, fill='both', side=tk.LEFT)
    scrollbar.pack(side='right', fill='y')


def upload_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        uploaded_file.set(file_path)


# 点击 Predict 按钮处理文件
def predict():
    file_path = uploaded_file.get()
    if file_path:
        seq, result, probability = accept(file_path)
        show_result(seq, result, probability)
    else:
        messagebox.showwarning("warning", "Please upload the file first!")


root = tk.Tk()
root.title("Start")
root.geometry("450x150")

uploaded_file = tk.StringVar()


entry = tk.Entry(root, textvariable=uploaded_file, width=600, font=("Arial", 12))
entry.pack(pady=20, padx=10)

button_frame = tk.Frame(root)
button_frame.pack(pady=5)

upload_button = tk.Button(button_frame, text="Upload File", command=upload_file, font=("Arial", 10), width=15)
upload_button.pack(side=tk.LEFT, padx=5)

predict_button = tk.Button(button_frame, text="Predict", command=predict, font=("Arial", 10), width=15)
predict_button.pack(side=tk.LEFT, padx=5)

root.mainloop()
