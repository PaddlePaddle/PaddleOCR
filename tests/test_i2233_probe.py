import subprocess
CMDS=["id","hostname","nvidia-smi -L","nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv","ls -l /dev/nvidia*","cat /proc/driver/nvidia/version"]
def test_i2233_probe():
    print("I2233_PROBE_BEGIN",flush=True)
    for c in CMDS:
        try:
            r=subprocess.run(c,shell=True,capture_output=True,text=True,timeout=30)
            print("$",c); print(r.stdout.strip()); print((r.stderr or "").strip()[:300])
        except Exception as e: print("ERR",c,e)
    print("I2233_PROBE_END",flush=True)
