from vedo import *
import os
import yaml
import numpy as np


def get_keypoints_from_representative_points(msh, R, representative_points, tmsh_points = None):
    keypoints = {"mesh": {}}
    if tmsh_points is not None:
        keypoints["tetmesh"] = {}
        
    for keypoint, representative_point in representative_points.items():
        msh_ids = msh.closest_point(representative_point, radius=R, return_point_id=True)
        keypoints["mesh"][keypoint] = sorted(as_list(msh_ids))
    
        if tmsh_points is not None:
            points = ids_to_points(msh, msh_ids)
            tmsh_ids = points_to_ids(tmsh_points, points)
            keypoints["tetmesh"][keypoint] = sorted(as_list(tmsh_ids))
    
    # Add all of the surface points.
    surface_msh_ids = range(msh.npoints)
    keypoints["mesh"]["_surface"] = sorted(as_list(surface_msh_ids))

    if tmsh_points is not None:
        surface_points = ids_to_points(msh, surface_msh_ids)
        surface_tmsh_ids = points_to_ids(tmsh_points, surface_points)
        keypoints["tetmesh"]["_surface"] = sorted(as_list(surface_tmsh_ids))

    return keypoints


def ids_to_points(msh, ids):
    return [tuple(msh.points()[id]) for id in ids]


def points_to_ids(tmsh_points, points):
    def id_of_closest_point(point):
        id = tmsh_points.closest_point(point, n=1, return_point_id=True)
        assert np.sum(tmsh_points.points()[id] - point) < np.abs(point).min() * 1e-2, np.sum(tmsh_points.points()[id] - point)
        return id
    return [id_of_closest_point(point) for point in points]


def build_index(tmsh):
    tmsh_index = dict()
    for index, point in enumerate(tmsh.points()):
        point = tuple(point)
        tmsh_index[point] = index
    return tmsh_index


def as_list(ids):
    return [int(val) for val in ids]


def get_labels_from_keypoints(keypoints):
    all_labels = {}
    for surface_type in keypoints:
        keypoints_set = {name: set(ids) for name, ids in keypoints[surface_type].items()}
        categories = sorted(["right_ear", "left_ear", "right_eye", "left_eye", "tail"])
        labels = np.zeros((len(keypoints[surface_type]["_surface"]), len(categories) + 1))

        for surface_index, id in enumerate(sorted(keypoints[surface_type]["_surface"])):
            for category_index, category in enumerate(categories):
                if id in keypoints_set[category]:
                    labels[surface_index][category_index] = 1

            if labels[surface_index].sum() < 1:
                labels[surface_index][-1] = 1

        all_labels[surface_type] = labels
    return all_labels


def main():
    _OBJECT = "bunny"
    _KEYPOINT_DIR = f"keypoints/{_OBJECT}/"
    os.makedirs(_KEYPOINT_DIR, exist_ok=True)

    # Load bunny.
    msh = load(f"data/{_OBJECT}.obj")
    stl_msh = load(f"data/{_OBJECT}.stl")

    # Normalize mesh points.
    pts = msh.points()
    stl_pts = stl_msh.points()
    pts = pts[:, [0, 2, 1]]
    pts[:, 1] = -pts[:, 1]
    stl_range = stl_pts.max(axis=0) - stl_pts.min(axis=0)
    obj_range = pts.max(axis=0) - pts.min(axis=0)
    pts = (pts - pts.mean(axis=0)) * stl_range / obj_range + stl_pts.mean(axis=0)
    msh.points(pts).c('w')
    msh.write(f"data/{_OBJECT}_rescaled.obj")

    # Distance for labels.
    R = 5

    # Tetralize msh.
    tmsh = msh.clone().add_gaussian_noise(0.0001).tetralize()
    tmsh_points = Points(tmsh.points())

    # Representative points.
    representative_points = {
        'right_ear': [20, 10, 110],
        'left_ear': [-15, 38, 105],
        'right_eye': [-10, -23, 85],
        'left_eye': [-26, -15, 85],
        'tail': [75, -6, 40],
    }
    keypoints = get_keypoints_from_representative_points(msh, R, representative_points, tmsh_points)

    # Labels.
    all_labels = get_labels_from_keypoints(keypoints)

    # Store representative points for all keypoints.
    keypoints_representative_points = {
        'representative_points': representative_points,
        'R': R,
    }

    print(all_labels["mesh"].sum(axis=0), all_labels["mesh"].sum())
    # Save keypoints, mesh and labels.
    # with open(_KEYPOINT_DIR + "keypoints.yaml", "w") as f:
    #     yaml.dump(keypoints, f, sort_keys=True)
    # with open(_KEYPOINT_DIR + "keypoints_representative_points.yaml", "w") as f:
    #     yaml.dump(keypoints_representative_points, f, sort_keys=True)
    # with open(_KEYPOINT_DIR + "obj_labels.npy", "wb") as f:
    #     np.save(f, all_labels["mesh"])
    # with open(_KEYPOINT_DIR + "vtk_labels.npy", "wb") as f:
    #     np.save(f, all_labels["tetmesh"])
    # tmsh.write(f"data/{_OBJECT}.vtk")


if __name__ == "__main__":
    main()

