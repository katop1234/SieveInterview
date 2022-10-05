import math

'''Creates tracker object that allows for tracking across frames'''

class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}

        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

        # Number of frames an id is seen for
        self.num_frames_seen_for = {}

    def replace(self, box_coords, id_to_replace):
        '''
        To use when we identify an object of another type with an existing id. i.e.
        if id 5 was referee but now it's other

        :param box_coords: [x1, y1, x2, y2]
        :param id_to_replace: id of the object we're replacing since its type has changed for some reason
        :return: new id of this object (an int)
        '''
        x1, y1, x2, y2 = box_coords
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # remove existing id
        del self.center_points[id_to_replace]

        # and replace it with a new id
        new_id = self.id_count
        self.center_points[new_id] = (cx, cy)
        self.id_count += 1
        self.num_frames_seen_for[new_id] = 1
        return new_id

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x1 ,y1, x2, y2, id])
                    same_object_detected = True
                    self.num_frames_seen_for[id] += 1
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                self.num_frames_seen_for[self.id_count] = 1

                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids


