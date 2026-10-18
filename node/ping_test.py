import subprocess, time
ECS="http://47.99.131.12:8081"; KEY="k0u4mail"
msg="PING_OK_%s"%time.strftime("%H%M%S")
open("/mnt/workspace/node1/ping.txt","w").write(msg)
r=subprocess.run(["curl","-s","-X","POST","%s/upload?name=ping_test.txt&key=%s"%(ECS,KEY),
                  "--data-binary",msg.encode()],capture_output=True,text=True,timeout=20)
print("msg=%s upload_rc=%s upload_out=%s"%(msg,r.returncode,r.stdout[:100]))
