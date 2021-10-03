import os


class UnifiedFileSystem:
    def __init__(self):
        pass

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

        raise NotImplementedError()
