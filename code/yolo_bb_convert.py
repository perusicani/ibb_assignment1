# ground truths format: YOLO!
# Class Label: 0 (or any integer representing the class label).
# Center X-coordinate: 0.7725 (normalized relative to the width of the image).
# Center Y-coordinate: 0.1719 (normalized relative to the height of the image).
# Width: 0.0225 (normalized relative to the width of the image).
# Height: 0.0547 (normalized relative to the height of the image).

def yolo_to_bbox(yolo_format, w, h):
    class_label, center_x, center_y, width, height = yolo_format
    
    # Convert YOLO format to absolute coordinates
    absolute_x = int(center_x * w)
    absolute_y = int(center_y * h)
    absolute_width = int(width * w)
    absolute_height = int(height * h)

    # Calculate top-left and bottom-right coordinates
    x1 = absolute_x - (absolute_width // 2)
    y1 = absolute_y - (absolute_height // 2)
    x2 = x1 + absolute_width
    y2 = y1 + absolute_height

    return (x1, y1, x2, y2)

def classifier_to_bbox(x, y, w, h):
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return (x1, y1, x2, y2)