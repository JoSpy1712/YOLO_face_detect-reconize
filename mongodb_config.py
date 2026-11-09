# Document the purpose of this module, class, or function.
"""Load local MongoDB settings without overriding the process environment."""
# Build filesystem paths relative to this script rather than the current working directory.
from pathlib import Path
# Read process environment settings and operating-system information.
import os


# Load only MONGODB_URI from the neighboring .env file without replacing an existing environment value.
def load_mongodb_env():
    # Set path from Path(__file__).resolve().parent / '.env'.
    path = Path(__file__).resolve().parent / '.env'
    # Continue with this branch only when not path.is_file().
    if not path.is_file():
        # Leave this function without a result.
        return
    # Process each line from path.read_text(encoding='utf-8').splitlines().
    for line in path.read_text(encoding='utf-8').splitlines():
        # Set (key, separator, value) from line.strip().partition('=').
        key, separator, value = line.strip().partition('=')
        # Continue with this branch only when separator and key.strip() == 'MONGODB_URI'.
        if separator and key.strip() == 'MONGODB_URI':
            # Set value from value.strip().
            value = value.strip()
            # Continue with this branch only when len(value) >= 2 and value[0] == value[-1] and (value[0] in '"\'').
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                # Set value from value[1:-1].
                value = value[1:-1]
            # URI characters stay literal: no shell execution or interpolation.
            # Call os.environ.setdefault with the arguments below.
            os.environ.setdefault('MONGODB_URI', value)
            # Leave this function without a result.
            return
