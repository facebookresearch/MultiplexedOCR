# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from virtual_fs import virtual_path
from virtual_fs.unified_fs import UnifiedFileSystem

# overwrite default behaviors
path = virtual_path

ufs = UnifiedFileSystem()
access = ufs.access
listdir = ufs.listdir
makedirs = ufs.makedirs
rmdir = ufs.rmdir
remove = ufs.remove


def system(command):
    if command.startswith("chmod"):
        print("[Warning] Skipped chmod for virtual os.system")
        return -1
    return os.system(command)


# inherit default behaviors
environ = os.environ
getpid = os.getpid
getuid = os.getuid
fspath = os.fspath
stat = os.stat

F_OK = os.F_OK
R_OK = os.R_OK
W_OK = os.W_OK
X_OK = os.X_OK
