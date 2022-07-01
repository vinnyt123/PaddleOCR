def map_res_to_selectext_format(res):
    print(res)
    
    out = []
    for el in res:
        word = el[0]
        bounding_box_initial = el[1]
        bounding_box = [{"x": bound[0], "y": bound[1]} for bound in bounding_box_initial]
        out.append({
            "text": word,
            "boundingBox": bounding_box,
        })
    
    print(out)
    return out