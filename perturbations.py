from vedo import load, Mesh
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

_NUM_PERTURBATIONS = 200
_OBJECT = "bunny"
_KEYPOINT_DIR = f"keypoints/{_OBJECT}/"

def perturb(msh, num_perturbations, labels):
    points = msh.points().tolist()
    faces = msh.faces()
    face_indices = np.random.choice(len(faces), num_perturbations)
    num_points = len(points)
    for i, f in enumerate(face_indices):
        face = faces[f]
        labels.append(np.min([labels[f] for f in face]))
        points.append(np.mean([points[f] for f in face], axis=0))
        for j in range(3):
            faces.append( [face[k] for k in range(3) if k != j] + [i + num_points])

    faces = [f for f in faces if f not in face_indices]

    newmsh = Mesh([points, faces])
    return newmsh, labels


def main():
    for i in range(20):
        msh = load("bunny_data/train/bunny.obj")
        labels = np.load('bunny_data/train/bunny_labels.npy')
        labels = np.argmax(labels, axis=1).tolist()

        newmsh, newlabels = perturb(msh, _NUM_PERTURBATIONS, labels)
        hotlabels = np.zeros((len(newlabels), np.max(newlabels) + 1))
        hotlabels[np.arange(len(newlabels)), newlabels] = 1


        newmsh.write(_KEYPOINT_DIR + f'perturb_bunny_{i}.obj')
        with open(_KEYPOINT_DIR + f"perturb_labels_{i}.npy", "wb") as f:
            np.save(f, hotlabels)


if __name__ == '__main__':
    main()