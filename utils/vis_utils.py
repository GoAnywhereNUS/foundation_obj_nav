import cv2
import numpy as np

class Visualiser:
    def __init__(self):
        self.vis_image_dims = (900, 1200)
        self.layouts = [
            { # for observations with 2 views
                'dims': (900, 1200),
                'processed0': (560, 25, 20), # width, start_row, start_col
                'processed1': (560, 475, 20),
                'live': (560, 25, 620), 
                'graph': (560, 420, 475, 620), # width, height, start_row, start_col
            },
            { # for observations with 2 < views <= 4
                'dims': (720, 1280),
                'processed0': (400, 40, 20),
                'processed1': (400, 40, 440),
                'processed2': (400, 380, 20),
                'processed3': (400, 380, 440),
                'live': (400, 40, 860),
                'graph': (400, 300, 380, 860),
            },
        ]
        self.curr_layout = None
        self.num_views = None

    def visualise_obs(self, obs, img_lang_obs, subgoal=None):
        locations = img_lang_obs['location']
        objects = img_lang_obs['object']

        # Choose layout
        if obs is not None:
            self.curr_layout = self.layouts[1]
            self.num_views = 4
            if len(obs) <= 2:
                self.curr_layout = self.layouts[0]
                self.num_views = 2
            elif len(obs) > 4:
                print("Only support 4 sensors! Using first 4 views.")
        else:
            return None

        images = []
        font, font_scale, font_thickness, font_colour = (
            cv2.FONT_HERSHEY_SIMPLEX, 1, 2, (20, 20, 200)
        )
        for view in obs.keys():
            rgb = obs[view]
            cam_id = view.split("_")[0]
            view_obj_bboxes, view_obj_labels, _ = objects[cam_id]
            view_location = locations[cam_id]

            cv2.putText(
                rgb, view_location, (10, 10),
                font, font_scale, font_colour, font_thickness, cv2.LINE_AA,
            )

            for bbox, label in zip(view_obj_bboxes, view_obj_labels):
                x0, y0, x1, y1 = bbox
                cv2.rectangle(
                    rgb, 
                    (int(x0), int(y0)), 
                    (int(x1), int(y1)), 
                    color=(0,0,0), 
                    thickness=1
                )
                cv2.putText(
                    rgb,
                    label,
                    (int(x0), int(y0) - 10),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.6,
                    color = (255, 255, 255),
                    thickness=2
                )

            if subgoal is not None and subgoal[0] == cam_id:
                _, (min_x, min_y, max_x, max_y) = subgoal
                cv2.rectangle(
                    rgb,
                    (int(min_x), int(min_y)),
                    (int(max_x), int(max_y)),
                    color=(20, 20, 255),
                    thickness=3
                )

            images.append((cam_id, rgb))

        # Layout
        if len(images) == 0:
            vis_image = np.ones((480, 640, 3)).astype(np.uint8) * 255

        else:
            iters = self.num_views
            vis_image_dims = self.curr_layout['dims']
            vis_image = np.ones((
                vis_image_dims[0], vis_image_dims[1], 3
            )).astype(np.uint8) * 255

            # Laying out processed observations
            for i in range(iters):
                cam_id, im = images[i]
                image_dims = im.shape
                width, start_row, start_col = self.curr_layout['processed' + str(i)]
                height = int(width / (image_dims[1] / image_dims[0]))
                vis_image[
                    start_row : start_row+height,
                    start_col : start_col+width
                ] = cv2.resize(im[0], (height, width))
                cv2.putText(
                    vis_image, cam_id, (start_row-20, start_col+(image_dims[1] // 2)),
                    font, font_scale, font_colour, font_thickness, cv2.LINE_AA,
                )

            # Laying out live stream
            width, start_row, start_col = self.curr_layout['live']
            height = int(width / (image_dims[1] / image_dims[0]))
            vis_image[
                start_row : start_row+height,
                start_col : start_col+width
            ] = cv2.resize(obs[obs.keys()[0]], (height, width))

        return vis_image
    

    def visualise_scene_graph(self, vis_image, scene_graph):
        # TODO: Lay out scene graph onto the vis_image