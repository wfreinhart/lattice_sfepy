import subprocess

proc = subprocess.Popen("python run_sfepy.py linear_elasticity.py", shell=True)
out, err = proc.communicate()

