import tkinter as tk
from tkinter import ttk
import queue
from datetime import datetime

class ToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(500, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background="#ffffe0", relief='solid', borderwidth=1,
                       font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()

class StatusWindow:
    def __init__(self, msg_queue: queue.Queue):
        self.q = msg_queue
        
        self.root = tk.Tk()
        self.root.title("Monitor do Recomendador de Áudio")
        self.root.geometry("600x450")
        self.root.configure(bg="#1e1e1e")
        
        # Tema Escuro e Fontes
        self.bg_color = "#1e1e1e"
        self.fg_color = "#ffffff"
        self.font_normal = ("Helvetica", 10)
        self.font_large = ("Helvetica", 14, "bold")
        self.font_xlarge = ("Helvetica", 16, "bold")
        
        # Variáveis de Estado
        self.always_on_top = tk.BooleanVar(value=True)
        self.root.attributes('-topmost', True)
        
        self.build_ui()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Iniciar polling da fila
        self.poll_queue()

    def build_ui(self):
        # Explicação fixa
        lbl_expl = tk.Label(self.root, text="O sistema escuta o MPD, analisa o som da faixa atual e adiciona à fila a música mais parecida",
                            bg=self.bg_color, fg="#aaaaaa", font=("Helvetica", 9, "italic"), wraplength=580)
        lbl_expl.pack(pady=5, padx=10, fill='x')
        
        # Status
        self.lbl_status = tk.Label(self.root, text="Inicializando...", font=self.font_xlarge, bg=self.bg_color, fg="#aaaaaa")
        self.lbl_status.pack(pady=10)
        ToolTip(self.lbl_status, "Indica o status atual do sistema e a conexão com o servidor MPD.")
        
        frame_info = tk.Frame(self.root, bg=self.bg_color)
        frame_info.pack(fill='x', padx=20, pady=5)
        
        # Tocando Agora
        tk.Label(frame_info, text="Tocando Agora:", bg=self.bg_color, fg="#888888", font=self.font_normal).grid(row=0, column=0, sticky='w', pady=2)
        self.lbl_track = tk.Label(frame_info, text="-", bg=self.bg_color, fg=self.fg_color, font=self.font_large)
        self.lbl_track.grid(row=0, column=1, sticky='w', pady=2, padx=10)
        ToolTip(self.lbl_track, "Música que o MPD está tocando no momento.")
        
        # Último Evento
        tk.Label(frame_info, text="Último Evento:", bg=self.bg_color, fg="#888888", font=self.font_normal).grid(row=1, column=0, sticky='w', pady=2)
        self.lbl_event = tk.Label(frame_info, text="-", bg=self.bg_color, fg=self.fg_color, font=self.font_normal)
        self.lbl_event.grid(row=1, column=1, sticky='w', pady=2, padx=10)
        ToolTip(self.lbl_event, "Último evento capturado do MPD (START, JUMP, etc).")
        
        # Última Recomendação
        tk.Label(frame_info, text="Recomendada:", bg=self.bg_color, fg="#888888", font=self.font_normal).grid(row=2, column=0, sticky='w', pady=2)
        self.lbl_rec = tk.Label(frame_info, text="-", bg=self.bg_color, fg="#88ccff", font=self.font_normal)
        self.lbl_rec.grid(row=2, column=1, sticky='w', pady=2, padx=10)
        ToolTip(self.lbl_rec, "Última música encontrada pela IA e se foi injetada com sucesso na fila.")
        
        # Contadores
        frame_counts = tk.Frame(self.root, bg=self.bg_color)
        frame_counts.pack(fill='x', padx=20, pady=10)
        
        self.lbl_cnt_rec = tk.Label(frame_counts, text="Recomendações: 0", bg=self.bg_color, fg="#44ff44", font=self.font_normal)
        self.lbl_cnt_rec.pack(side='left', padx=10)
        
        self.lbl_cnt_idx = tk.Label(frame_counts, text="Indexadas: 0", bg=self.bg_color, fg="#ffff44", font=self.font_normal)
        self.lbl_cnt_idx.pack(side='left', padx=10)
        
        self.lbl_cnt_err = tk.Label(frame_counts, text="Erros: 0", bg=self.bg_color, fg="#ff4444", font=self.font_normal)
        self.lbl_cnt_err.pack(side='left', padx=10)
        ToolTip(frame_counts, "Contadores de ações ocorridas nesta sessão.")
        
        # Log
        tk.Label(self.root, text="Log de Ações:", bg=self.bg_color, fg="#888888", font=self.font_normal).pack(anchor='w', padx=20)
        self.list_log = tk.Listbox(self.root, bg="#2d2d2d", fg="#ffffff", font=("Consolas", 9), height=8, relief='flat')
        self.list_log.pack(fill='both', expand=True, padx=20, pady=5)
        ToolTip(self.list_log, "Histórico das últimas operações do sistema.")
        
        # Checkbox Topmost
        cb_top = tk.Checkbutton(self.root, text="Manter no topo", variable=self.always_on_top, command=self.toggle_topmost,
                                bg=self.bg_color, fg=self.fg_color, selectcolor="#1e1e1e", activebackground=self.bg_color, activeforeground=self.fg_color)
        cb_top.pack(side='right', padx=20, pady=5)

    def toggle_topmost(self):
        self.root.attributes('-topmost', self.always_on_top.get())
        
    def add_log(self, msg: str):
        t = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{t}] {msg}"
        self.list_log.insert(tk.END, full_msg)
        if self.list_log.size() > 10:
            self.list_log.delete(0)
        self.list_log.yview(tk.END)

    def poll_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                self.process_message(msg)
        except queue.Empty:
            pass
        
        self.root.after(100, self.poll_queue)
        
    def process_message(self, msg: dict):
        msg_type = msg.get("type")
        
        if msg_type == "status":
            state = msg.get("state")
            if state == "monitorando":
                self.lbl_status.config(text="Monitorando...", fg="#44ff44")
            elif state == "indexando":
                self.lbl_status.config(text="Analisando faixa / indexando...", fg="#ffff44")
            elif state == "erro":
                self.lbl_status.config(text="Erro / MPD desconectado", fg="#ff4444")
                
        elif msg_type == "track":
            track_name = msg.get("name")
            self.lbl_track.config(text=track_name)
            
        elif msg_type == "event":
            event_name = msg.get("name")
            t = datetime.now().strftime("%H:%M:%S")
            self.lbl_event.config(text=f"{event_name} às {t}")
            
        elif msg_type == "recommendation":
            rec_text = msg.get("text")
            self.lbl_rec.config(text=rec_text)
            
        elif msg_type == "counters":
            recs = msg.get("recs", 0)
            idxs = msg.get("idxs", 0)
            errs = msg.get("errs", 0)
            self.lbl_cnt_rec.config(text=f"Recomendações: {recs}")
            self.lbl_cnt_idx.config(text=f"Indexadas: {idxs}")
            self.lbl_cnt_err.config(text=f"Erros: {errs}")
            
        elif msg_type == "log":
            self.add_log(msg.get("text"))

    def on_close(self):
        # Apenas destrói a janela e finaliza a mainloop.
        # Como a thread do MPD é daemon, o programa encerrará em seguida.
        self.root.destroy()
        
    def start(self):
        self.root.mainloop()
