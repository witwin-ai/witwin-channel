# Copyright Xingyu Chen.
# Runs `python -m witwin.channel doctor`, the installation self-check.

import sys

from witwin.channel.deployment import main

sys.exit(main())
