from PIL import Image

from laya_omni.data import tile_views


def test_tile_views_whole_image_then_crops_in_reading_order():
    img = Image.new("RGB", (100, 60))
    img.putpixel((99, 59), (255, 0, 0))  # bottom-right corner
    views = tile_views(img, 2)
    assert len(views) == 5 and views[0] is img
    assert [v.size for v in views[1:]] == [(50, 30)] * 4
    assert views[4].getpixel((49, 29)) == (255, 0, 0)


def test_no_tiles_keeps_the_image():
    img = Image.new("RGB", (10, 10))
    assert tile_views(img, 0) == [img] and tile_views(img, 1) == [img]
