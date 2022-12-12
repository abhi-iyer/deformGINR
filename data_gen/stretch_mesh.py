from vedo import *
import os
import yaml
import numpy as np
import generate_keypoints


def first_deformation(msh):
    sources = [[5, 38, 98], [60, -20, 40], [10, -20, 82]]    # this point moves
    targets = [[50, 38, 98], [80, -10, 40], [20, -20, 82]]  # ...to this.
    sources = np.asarray(sources)
    targets = np.asarray(targets)
    warp = msh.clone().warp(sources, targets)
    warp.c("blue", 0.3)
    show(msh, warp, axes=1).close()

    # Save warped mesh.
    warp.write(f"data/bunny_rescaled_deformed_1.obj")


def first_deformation_subdivided(msh):
    # Subdivide mesh.
    mshsub = msh.clone().triangulate().subdivide(n=2)

    # Load original representative points for keypoints.
    _KEYPOINT_DIR = "keypoints/bunny/"
    with open(_KEYPOINT_DIR + "keypoints_representative_points.yaml", "r") as f:
        keypoints_representative_points = yaml.safe_load(f)
    
    # Find new keypoints.
    representative_points = keypoints_representative_points["representative_points"]
    R = keypoints_representative_points["R"]
    keypoints = generate_keypoints.get_keypoints_from_representative_points(mshsub, R, representative_points)
    all_labels = generate_keypoints.get_labels_from_keypoints(keypoints)

    # Transform mesh.
    sources = [[5, 38, 98], [60, -20, 40], [10, -20, 82]]    # this point moves
    targets = [[50, 38, 98], [80, -10, 40], [20, -20, 82]]   # ...to this.
    sources = np.asarray(sources)
    targets = np.asarray(targets)
    warp = mshsub.clone().warp(sources, targets)
    warp.c("blue", 0.3)
    show(mshsub, warp, axes=1).close()

    # Save warped mesh.
    warp.write(f"data/bunny_rescaled_deformed_subdivided_1.obj")

    # Save new labels.
    with open(_KEYPOINT_DIR + "bunny_rescaled_deformed_subdivided_1_keypoints.yaml", "w") as f:
        yaml.dump(keypoints, f, sort_keys=True)
    with open(_KEYPOINT_DIR + "bunny_rescaled_deformed_subdivided_1_labels.npy", "wb") as f:
        np.save(f, all_labels["mesh"])


def first_deformation_decimated(msh):
    # Decimate mesh.
    mshdec = msh.clone().triangulate().decimate(n=1000)

    # Load original representative points for keypoints.
    _KEYPOINT_DIR = "keypoints/bunny/"
    with open(_KEYPOINT_DIR + "keypoints_representative_points.yaml", "r") as f:
        keypoints_representative_points = yaml.safe_load(f)
    
    # Find new keypoints.
    representative_points = keypoints_representative_points["representative_points"]
    R = keypoints_representative_points["R"]
    keypoints = generate_keypoints.get_keypoints_from_representative_points(mshdec, R, representative_points)
    all_labels = generate_keypoints.get_labels_from_keypoints(keypoints)

    # Transform mesh.
    sources = [[5, 38, 98], [60, -20, 40], [10, -20, 82]]    # this point moves
    targets = [[50, 38, 98], [80, -10, 40], [20, -20, 82]]   # ...to this.
    sources = np.asarray(sources)
    targets = np.asarray(targets)
    warp = mshdec.clone().warp(sources, targets)
    warp.c("blue", 0.3)
    show(mshdec, warp, axes=1).close()

    # Save warped mesh.
    warp.write(f"data/bunny_rescaled_deformed_decimated_1.obj")

    # Save new labels.
    with open(_KEYPOINT_DIR + "bunny_rescaled_deformed_decimated_1_keypoints.yaml", "w") as f:
        yaml.dump(keypoints, f, sort_keys=True)
    with open(_KEYPOINT_DIR + "bunny_rescaled_deformed_decimated_1_labels.npy", "wb") as f:
        np.save(f, all_labels["mesh"])


def second_deformation(msh):
    sources = [[10, 38, 98], [60, -20, 40], [10, -20, 82]]    # this point moves
    targets = [[50, 98, 98], [80, -40, 40], [20, -20, 62]]  # ...to this.
    sources = np.asarray(sources)
    targets = np.asarray(targets)
    warp = msh.clone().warp(sources, targets)
    warp.c("blue", 0.3)
    show(msh, warp, axes=1).close()

    # Save warped mesh.
    warp.write(f"data/bunny_rescaled_deformed_2.obj")


def main():
    settings.use_depth_peeling = True

    # Load bunny.
    msh = load(f"data/bunny_rescaled.obj")

    # Create deformations.
    first_deformation(msh.clone())
    second_deformation(msh.clone())
    first_deformation_subdivided(msh.clone())
    first_deformation_decimated(msh.clone())



if __name__ == "__main__":
    main()

