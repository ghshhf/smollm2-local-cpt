# -*- coding: utf-8 -*-
"""临时文件上传接收服务（一次性传输用，收完可关）
POST /upload 保存到 D:/work/novel/inbox/
GET  /        上传页面
"""
import os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

INBOX = r"D:\work\novel\inbox"
PORT = 8899
os.makedirs(INBOX, exist_ok=True)

PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>上传到训练机</title>
<style>body{font-family:sans-serif;max-width:640px;margin:60px auto;padding:0 20px}
.drag{border:3px dashed #999;border-radius:16px;padding:80px 20px;text-align:center;font-size:20px;color:#666}
.drag.over{border-color:#2b7;color:#2b7;background:#f0faf4}
#log{margin-top:20px;color:#333}</style></head><body>
<h2>📁 上传小说合集 → 训练机</h2>
<div class="drag" id="drop">把文件拖到这里，或点击选择<br><small>支持 zip / txt / epub / json 等，可多选</small></div>
<input type="file" id="file" multiple style="display:none">
<div id="log"></div>
<script>
const drop=document.getElementById('drop'), fi=document.getElementById('file'), log=document.getElementById('log');
drop.onclick=()=>fi.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');send(e.dataTransfer.files)};
fi.onchange=()=>send(fi.files);
function send(files){
  for(const f of files){
    const fd=new FormData(); fd.append('file',f,f.name);
    log.innerHTML+='\\n⬆️ 上传中: '+f.name+' ('+(f.size/1048576).toFixed(1)+' MB)...';
    fetch('/upload',{method:'POST',body:fd}).then(r=>r.text()).then(t=>{
      log.innerHTML+='\\n✅ 完成: '+f.name;
    }).catch(e=>log.innerHTML+='\\n❌ 失败: '+f.name+' '+e);
  }
}
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))
    def do_GET(self):
        if self.path.startswith("/ok"):
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        import traceback
        try:
            self._handle_upload()
        except Exception:
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            try:
                self.send_response(500); self.end_headers(); self.wfile.write(b"error")
            except Exception:
                pass
    def _handle_upload(self):
        if self.path != "/upload":
            self.send_response(404); self.end_headers(); return
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            self.send_response(400); self.end_headers(); return
        # 手动读 multipart（避免 cgi 模块 py3.13 移除问题）
        import re
        boundary = ctype.split("boundary=")[-1].encode()
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)
        parts = data.split(b"--" + boundary)
        saved = []
        for p in parts:
            if b'Content-Disposition' not in p: continue
            m = re.search(rb'filename="([^"]+)"', p)
            if not m: continue
            fname = os.path.basename(m.group(1).decode("utf-8", "ignore"))
            body = p.split(b"\r\n\r\n", 1)
            if len(body) < 2: continue
            content = body[1].rsplit(b"\r\n", 1)[0]
            # 去掉结尾可能的 --
            if content.endswith(b"--"): content = content[:-2]
            out = os.path.join(INBOX, fname)
            with open(out, "wb") as f: f.write(content)
            saved.append((fname, len(content)))
        msg = "saved: " + ", ".join(f"{n}({s//1024}KB)" for n,s in saved)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(msg.encode())

if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    print(f"upload server on :{PORT}, inbox={INBOX}", flush=True)
    srv.serve_forever()
