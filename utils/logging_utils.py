
"""
Utilities for recording debug information
"""

from PIL import Image, ImageDraw, ImageFont

def draw_annotated_obs(views, obs):
    """
    Input: obs, observations from environment
           views, output of parseImage with structure,
    {
        "view": { 
            "place" : ...,
            "objects" : {
                "class1": [(label, attr, bbox, im_crop), ...],
                "class2": ...
            }
        }, ...
    }
    """
    for view, view_data in views.items():
        print("##", view)
        image = Image.fromarray(obs[view]['image'])
        draw = ImageDraw.Draw(image)

        for cls, objects in view_data['objects'].items():
            for obj in objects:
                obs_id, label, attr, bbox, image_crop, nidxs = obj
                draw_colour = "green" if cls.lower() == "connector" else "red"
                draw.rectangle(bbox, outline=draw_colour, width=2)

                # Draw label and attr
                text = f"{obs_id}: {label} | {attr} | {nidxs}"
                text_position = (bbox[0], bbox[1] + 10)
                draw.text(text_position, text, fill=draw_colour)

                image_crop.save(f'imgs/{label}_{attr}.png')

        place_class = view_data["place_class"]
        place_label = view_data['place']
        place_text = f"{place_class} | {place_label}"
        draw.text((10, 10), place_label, fill="white")
        image.save(f'imgs/{view}.png')
        