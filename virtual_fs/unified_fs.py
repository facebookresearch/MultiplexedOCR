import logging
import os
import random

logger = logging.getLogger(__name__)


class UnifiedFileSystem:
    def __init__(self):
        pass

    def access(self, path, mode):
        path_struct = self.get_path_struct(path)

        if "local" in path_struct:
            return os.access(path, mode)

        raise NotImplementedError(f"unrecognized path_struct: {path_struct}")
        
    def copy2(self, src, dst):
        logger.info(f"Copying {src} to {dst}")
        path_struct_src = self.get_path_struct(src)
        path_struct_dst = self.get_path_struct(dst)
        if "local" in path_struct_src:
            if "local" in path_struct_dst:
                return shutil.copy2(path_struct_src["local"], path_struct_dst["local"])
            else:
                raise NotImplementedError(f"unrecognized path_struct: {path_struct_dst}")
        else:
            raise NotImplementedError(f"unrecognized path_struct: {path_struct_src}")
            
    def copytree(self, src, dst, symlinks=False, ignore=None):
        logger.info(f"Copying {src} to {dst}")
        path_struct_src = self.get_path_struct(src)
        path_struct_dst = self.get_path_struct(dst)

        if "local" in path_struct_src:
            if "local" in path_struct_dst:
                return shutil.copytree(
                    src=path_struct_src["local"],
                    dst=path_struct_dst["local"],
                    symlinks=symlinks,
                    ignore=ignore,
                )
            else:
                raise NotImplementedError(f"unrecognized path_struct: {path_struct_dst}")
        else:
            raise NotImplementedError(f"unrecognized path_struct: {path_struct_src}")

    def exists(self, path=""):
        path_struct = self.get_path_struct(path)

        if "local" in path_struct:
            return os.path.exists(path_struct["local"])

        return False

    def isdir(self, path=""):
        path_struct = self.get_path_struct(path)
        if "local" in path_struct:
            return os.path.isdir(path_struct["local"])

        return False

    def isfile(self, path=""):
        path_struct = self.get_path_struct(path)

        if "local" in path_struct:
            return os.path.isfile(path_struct["local"])

        return False

    def get_path_struct(self, path):
        if path.startswith("~"):
            path = os.path.expanduser(path)

        return {"local": path}
    
    def get_random_str(self, length):
        # get random string without repeating letters
        return "".join(random.sample(string.ascii_lowercase, length))

    def listdir(self, path=""):
        path_struct = self.get_path_struct(path)

        if "local" in path_struct:
            return os.listdir(path_struct["local"])

        entries = []

        return entries

    def open(self, path="", mode="r", encoding=None):
        path_struct = self.get_path_struct(path)
        if "local" in path_struct:
            return open(path_struct["local"], mode=mode, encoding=encoding)

        raise NotImplementedError(f"unrecognized path_struct: {path_struct}")

    def makedirs(self, path, exist_ok=False):
        path_struct = self.get_path_struct(path)

        if "local" in path_struct:
            logger.info("Making local dir {}".format(path_struct["local"]))
            return os.makedirs(name=path_struct["local"], exist_ok=exist_ok)

        raise NotImplementedError(f"unrecognized path_struct: {path_struct}")
    
    def mkdtemp(self, prefix):
        path = prefix + self.get_random_str(6)
        self.makedirs(path)
        return path

    def remove(self, path):
        path_struct = self.get_path_struct(path)
        if "local" in path_struct:
            return os.remove(path)

        raise NotImplementedError(f"unrecognized path_struct: {path_struct}")

    def rmdir(self, path):
        path_struct = self.get_path_struct(path)
        if "local" in path_struct:
            return os.rmdir(path)

        raise NotImplementedError(f"unrecognized path_struct: {path_struct}")

    def rmtree(self, path):
        path_struct = self.get_path_struct(path)

        if "local" in path_struct:
            return shutil.rmtree(path)

        raise NotImplementedError(f"unrecognized path_struct: {path_struct}")