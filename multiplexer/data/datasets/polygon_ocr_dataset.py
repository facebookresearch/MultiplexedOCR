import logging
import math

from PIL import Image, ImageDraw, ImageFont

from virtual_fs import virtual_os as os
from virtual_fs.virtual_io import open

logger = logging.getLogger(__name__)


class PolygonOcrDataset(object):
    def __init__(self, name, transforms=None):
        self.name = name
        self.transforms = transforms
        self.font_paths = {
            "default": "/path/to/Arial Unicode.ttf",
        }
        self.fonts = {}

    def visualize(self, img, target):
        if self.transforms is not None:
            from torchvision import transforms as T

            img = T.ToPILImage()(img)

        img = img.convert("RGBA")

        text_color = (255, 0, 0, 255)
        polygon_list = target.get_field("masks").polygons
        chars_boxes = target.get_field("char_masks").chars_boxes
        languages = target.get_field("languages")
        num_boxes = len(polygon_list)
        assert len(chars_boxes) == num_boxes
        assert len(languages) == num_boxes
        annotation_img = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(annotation_img)
        for (polygon_item, chars_box, language) in zip(polygon_list, chars_boxes, languages):
            polygon = polygon_item.polygons[0].tolist()
            draw.polygon(polygon, outline=(0, 255, 0, 128), fill=(0, 255, 0, 64))
            lang_word = "{} ({})".format(chars_box.word, language)

            font_key = language if language in self.font_paths else "default"

            if font_key not in self.fonts:
                font_path = self.font_paths[font_key]
                if os.path.exists(font_path):
                    with open(font_path, "rb") as f:
                        self.fonts[font_key] = ImageFont.truetype(f, 15)
                else:
                    logger.warning("Font {} doesn't exist, using default.".format(font_path))
                    self.fonts[font_key] = ImageFont.load_default()

            draw.text(
                xy=(polygon[0], polygon[1]),
                text=lang_word,
                font=self.fonts[font_key],
                fill=text_color,
            )
        return Image.alpha_composite(img, annotation_img)

    def visualize_rotated_boxes(self, img, target):
        if self.transforms is not None:
            from torchvision import transforms as T

            img = T.ToPILImage()(img)

        img = img.convert("RGBA")

        text_color = (255, 0, 0, 255)
        polygon_list = target.get_field("masks").polygons
        chars_boxes = target.get_field("char_masks").chars_boxes
        languages = target.get_field("languages")
        rotated_boxes = target.get_field("rotated_boxes_5d")
        num_boxes = len(polygon_list)
        assert len(chars_boxes) == num_boxes
        assert len(languages) == num_boxes
        annotation_img = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(annotation_img)
        for (rotated_box, chars_box, language) in zip(rotated_boxes, chars_boxes, languages):
            cnt_x, cnt_y, w, h, angle = rotated_box.tolist()
            theta = angle * math.pi / 180.0
            c = math.cos(theta)
            s = math.sin(theta)
            rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
            # x: left->right ; y: top->down
            rotated_rect = [(s * yy + c * xx + cnt_x, c * yy - s * xx + cnt_y) for (xx, yy) in rect]
            for k in range(4):
                j = (k + 1) % 4
                draw.line(
                    xy=[
                        rotated_rect[k][0],
                        rotated_rect[k][1],
                        rotated_rect[j][0],
                        rotated_rect[j][1],
                    ],
                    fill=(255, 0, 0, 128) if k == 1 else (0, 255, 0, 128),
                    width=3,
                )

            # draw.polygon(polygon, outline=(0, 255, 0, 128), fill=(0, 255, 0, 64))
            lang_word = "{} ({})".format(chars_box.word, language)

            font_key = language if language in self.font_paths else "default"

            if font_key not in self.fonts:
                font_path = self.font_paths[font_key]
                if os.path.exists(font_path):
                    with open(font_path, "rb") as f:
                        self.fonts[font_key] = ImageFont.truetype(f, 15)
                else:
                    logger.warning("Font {} doesn't exist, using default.".format(font_path))
                    self.fonts[font_key] = ImageFont.load_default()

            draw.text(
                xy=[rotated_rect[0][0], rotated_rect[0][1]],
                text=lang_word,
                font=self.fonts[font_key],
                fill=text_color,
            )
        return Image.alpha_composite(img, annotation_img)
