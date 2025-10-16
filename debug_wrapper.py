import sys
import urllib.request
from run_vcc import main, parse_arguments

# 1. Store the original internet function just in case
original_urlopen = urllib.request.urlopen

# 2. Define our "trap" function
def no_internet_urlopen(*args, **kwargs):
    """
    This function replaces the real urlopen.
    It will be called by any library that tries to access the internet.
    """
    url = args[0].full_url if hasattr(args[0], 'full_url') else args[0]
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!                NETWORK ACCESS ATTEMPT CAUGHT              !!!")
    print(f"!!! A library tried to connect to the following URL:        !!!")
    print(f"!!! {url}")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # Raise a very specific error so we know our trap worked.
    raise ConnectionError("NETWORK ACCESS IS FORBIDDEN BY THE DEBUG WRAPPER.")

# 3. MONKEYPATCH: Replace the real function with our trap
urllib.request.urlopen = no_internet_urlopen

print("--- Debug Wrapper Activated: Internet access has been disabled. ---")
print("--- Now running the main VCC script... ---")

# 4. Run the original script's main function
if __name__ == '__main__':
    # Pass along all the command-line arguments
    args = parse_arguments(sys.argv[1:])
    main(args)
