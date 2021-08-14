# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import glob
import os
import re
import zipfile


class MLT19Utils:
    # Handles validation dataset only for now
    num_images_per_language = 200
    language_list = ["ar", "en", "fr", "zh", "de", "ko", "ja", "it", "bn", "hi"]

    @classmethod
    def get_language_from_filename(cls, file_name):
        # e.g., res_img_01000.txt -> res_img_01000
        name = file_name.split(".")[-2]
        # e.g., res_img_01000 -> 1000
        id = int(name.split("_")[-1])
        # e.g., 1000 -> 4, 1001 -> 5
        lang_id = (id - 1) // cls.num_images_per_language
        # e.g., language_list[4] == "de"
        return cls.language_list[lang_id]

    @classmethod
    def get_result_file_list(cls, results_dir, split="val", languages=None):
        assert os.path.exists(results_dir), f"Path {results_dir} does not exist!"
        file_list = glob.glob("{}/*.txt".format(results_dir))
        if languages is None:
            return file_list

        assert split == "val", f"Not available for split {split}"

        output_list = []

        for file_name in file_list:
            language = cls.get_language_from_filename(file_name)
            if language in languages:
                output_list.append(file_name)

        return output_list

    @classmethod
    def load_zip_file(cls, file, fileNameRegExp="", allEntries=False, languages=None):
        """
        Returns an array with the contents (filtered by fileNameRegExp) of a ZIP file.
        The key's are the names or the file or the capturing group definied in the fileNameRegExp
        allEntries validates that all entries in the ZIP file pass the fileNameRegExp
        """
        archive = zipfile.ZipFile(file, mode="r", allowZip64=True)

        pairs = []
        for name in archive.namelist():
            # example: gt_img_00001.txt
            keyName = name
            if fileNameRegExp != "":
                # example: <re.Match object; span=(0, 16), match='gt_img_00001.txt'>
                m = re.match(fileNameRegExp, name)
                if m is None:
                    if allEntries:
                        raise Exception("ZIP entry not valid: %s" % name)
                    continue
                else:
                    if len(m.groups()) > 0:
                        # example: 00001
                        keyName = m.group(1)

            if (languages is None) or (cls.get_language_from_filename(name) in languages):
                pairs.append([keyName, archive.read(name)])

        return dict(pairs)
