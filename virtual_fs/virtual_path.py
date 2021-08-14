# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from virtual_fs.unified_fs import UnifiedFileSystem

# overwrite default behavior
ufs = UnifiedFileSystem()

exists = ufs.exists
isdir = ufs.isdir
isfile = ufs.isfile

# inherit default behavior
abspath = os.path.abspath
basename = os.path.basename
dirname = os.path.dirname
expanduser = os.path.expanduser
join = os.path.join
normpath = os.path.normpath
split = os.path.split
splitext = os.path.splitext
