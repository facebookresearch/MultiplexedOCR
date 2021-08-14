# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from virtual_fs.unified_fs import UnifiedFileSystem

ufs = UnifiedFileSystem()

mkdtemp = ufs.mkdtemp
