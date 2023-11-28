from datetime import datetime


def get_current_datetime():
    datetime_str = "[HERBALOOK LOG " + str(datetime.now()) + "]"
    return datetime_str
