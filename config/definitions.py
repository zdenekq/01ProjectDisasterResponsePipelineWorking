'''
Identify root directory ROOT_DIR on runtime
https://towardsdatascience.com/simple-trick-to-work-with-relative-paths-in-python-c072cdc9acb9

'''

import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

