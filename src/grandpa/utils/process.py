import psutil
import subprocess

def check_if_process_running(processName):
    '''
    Check if there is any running process that contains the given name processName.
    '''
    #Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def start_process_if_not_running(process_name, *args):
    if not check_if_process_running(process_name):
        subprocess.Popen([process_name, *args])
