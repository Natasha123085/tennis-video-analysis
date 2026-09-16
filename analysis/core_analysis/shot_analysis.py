def classify_shot(landing_position, zones):
    x, y = landing_position

    if x < zones["left"][1]:
        horizontal = "left"
    elif x < zones["middle"][1]:
        horizontal = "middle"
    else:
        horizontal = "right"

    if y < zones["short"][1]:
        depth = "short"
    elif y < zones["mid"][1]:
        depth = "mid"
    else:
        depth = "deep"

    return horizontal, depth
