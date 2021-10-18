from virtual_fs.unified_fs import UnifiedFileSystem

ufs = UnifiedFileSystem()

copy2 = ufs.copy2
copytree = ufs.copytree
rmtree = ufs.rmtree