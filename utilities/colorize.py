from enum import Enum


class Color(Enum):
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    MAGENTA = 5
    CYAN = 6
    WHITE = 7
    BLACK = 8


def _string_to_color(string):
    string = string.lower()
    return {
        "red": Color.RED,
        "green": Color.GREEN,
        "yellow": Color.YELLOW,
        "blue": Color.BLUE,
        "magenta": Color.MAGENTA,
        "cyan": Color.CYAN,
        "white": Color.WHITE,
        "black": Color.BLACK,
    }.get(string, Color.WHITE)


def color(text, is_bold=False, text_color="white", background_color="black"):
    """
    Colorizes text with ANSI escape codes.

    Parameters:
        text (str): The text to colorize.
        is_bold (bool): Whether the text should be bold.
        text_color (str): The color of the text.
        background_color (str): The color of the background.

    Returns:
        str: The colorized text.
    """
    res = "\33["
    if is_bold:
        res += "1;"
    res += f"3{_string_to_color(text_color).value};4{_string_to_color(background_color).value}m{text}\33[0m"
    return res


def rainbow(text, is_bold=False, background_color="black"):
    res = ""
    for i, char in enumerate(text):
        res += color(
            char,
            is_bold=is_bold,
            text_color=Color(i % 6 + 1).name.lower(),
            background_color=background_color,
        )
    return res


def __test_colors():
    for text_color in Color:
        for background_color in Color:
            print(
                color(
                    "Hello, world!",
                    text_color=text_color.name.lower(),
                    background_color=background_color.name.lower(),
                ),
                "Text Color:",
                color(text_color.name, text_color=text_color.name),
                "BG Color:",
                color(background_color.name, text_color=background_color.name),
            )
            print(
                color(
                    "Hello, world!",
                    is_bold=True,
                    text_color=text_color.name.lower(),
                    background_color=background_color.name.lower(),
                ),
                "Text Color:",
                color(text_color.name, text_color=text_color.name),
                "BG Color:",
                color(background_color.name, text_color=background_color.name),
            )
