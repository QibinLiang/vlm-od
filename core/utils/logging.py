# Copyright (c) 2025 Qibin Liang (RFAI tech) <physechan@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import logging

LOG_COLORS = {
    "DEBUG": "\033[37m",    
    "INFO": "\033[36m",     
    "NOTICE": "\033[34m",    
    "WARNING": "\033[33m",  
    "ERROR": "\033[31m",    
    "CRITICAL": "\033[41m",  
}

RESET = "\033[0m"
VERBOSE = 25
logging.addLevelName(VERBOSE, "VERBOSE")

class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = LOG_COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{RESET}"
        return super().format(record)

def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)

logging.Logger.verbose = verbose

def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    formatter = ColorFormatter(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(VERBOSE)
    root.handlers = []
    root.addHandler(handler)

def get_logger(name: str = None):
    return logging.getLogger(name)
