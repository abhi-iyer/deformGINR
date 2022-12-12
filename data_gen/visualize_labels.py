from vedo import *
import os
import yaml
import numpy as np
import torchmetrics
import torch


def get_colored_point_clouds(msh, labels, colormap):
    all_label_points = []
    for label, label_color in colormap.items():
        label_points = Points(msh.points()[np.where(labels == label)[0]], r=10).c(label_color)
        all_label_points.append(label_points)
    return all_label_points


def main():
    settings.use_depth_peeling = True
    _OBJECT = "bunny"
    view1 = (-200, 300, 120)
    view2 = (300, -300, 120)
    viewup = (-0.02, -0.02, 0.9)
    colormap = {
        0: '#1f77b4',
        1: '#ff7f0e',
        2: '#2ca02c',
        3: '#d62728',
        4: '#9467bd',
    }

    # Show predictions of deformed bunny.
    msh = load(f"data/{_OBJECT}_rescaled_deformed_1.obj").c('#ffffff')
    centre_of_mass = msh.points().mean(axis=0)
    preds = np.load("model_outputs/bunny_deformed_predictions.npy")
    assert len(preds.shape) == 1

    all_label_points = get_colored_point_clouds(msh, preds, colormap)
    plt = Plotter(axes=1, interactive=False)
    plt.show(msh, all_label_points, zoom=1.08, camera={'pos': view1, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot('images/bunny_deformed_predictions_view1.png')
    plt.show(msh, all_label_points, zoom=1.08, camera={'pos': view2, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot('images/bunny_deformed_predictions_view2.png')


    # Show predictions of original bunny.
    msh = load(f"data/{_OBJECT}_rescaled.obj").c('#ffffff')
    centre_of_mass = msh.points().mean(axis=0)
    preds = np.load("model_outputs/regular_bunny_predictions.npy")
    assert len(preds.shape) == 1

    all_label_points = get_colored_point_clouds(msh, preds, colormap)
    plt = Plotter(axes=1, interactive=False)
    plt.show(msh, all_label_points, zoom=1.25, camera={'pos': view1, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot('images/bunny_predictions_view1.png')
    plt.show(msh, all_label_points, zoom=1.25, camera={'pos': view2, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot('images/bunny_predictions_view2.png')


    # Show labels of original bunny.
    msh = load(f"data/{_OBJECT}_rescaled.obj").c('#ffffff')
    centre_of_mass = msh.points().mean(axis=0)
    labels = np.load("keypoints/bunny/obj_labels.npy")
    assert len(labels.shape) == 2
    assert np.all(np.sum(labels, axis=1) == 1.)

    labels = np.argmax(labels, axis=1)

    all_label_points = get_colored_point_clouds(msh, labels, colormap)
    plt = Plotter(axes=1, interactive=False)
    plt.show(msh, all_label_points, zoom=1.25, camera={'pos': view1, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot('images/bunny_labelled_view1.png')
    plt.show(msh, all_label_points, zoom=1.25, camera={'pos': view2, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot('images/bunny_labelled_view2.png')


    # Show labels of deformed and subdivided bunny.
    msh = load(f"data/{_OBJECT}_rescaled_deformed_subdivided_1.obj").c('#ffffff')
    # orig_msh = load(f"data/{_OBJECT}_rescaled.obj").c('#F9F6A6').wireframe()
    centre_of_mass = msh.points().mean(axis=0)
    labels = np.load("keypoints/bunny/bunny_rescaled_deformed_subdivided_1_labels.npy")
    assert len(labels.shape) == 2
    assert np.all(np.sum(labels, axis=1) == 1.)

    labels = np.argmax(labels, axis=1)

    all_label_points = get_colored_point_clouds(msh, labels, colormap)
    plt = Plotter(axes=1, interactive=False)
    plt.show(msh, all_label_points, zoom=1.08, camera={'pos': view1, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot('images/bunny_deformed_subdivided_labelled_view1.png')
    plt.show(msh, all_label_points, zoom=1.08, camera={'pos': view2, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot('images/bunny_deformed_subdivided_labelled_view2.png')


    # Show predictions of deformed and subdivided bunny.
    msh = load(f"data/{_OBJECT}_rescaled_deformed_subdivided_1.obj").c('#ffffff')
    centre_of_mass = msh.points().mean(axis=0)
    preds = np.load("model_outputs/bunny_oversampled_deformed.npy")
    assert len(preds.shape) == 1

    # print(torchmetrics.Accuracy(num_classes=6, average=None, multiclass=True)(torch.IntTensor(preds), torch.IntTensor(labels)))

    all_label_points = get_colored_point_clouds(msh, preds, colormap)
    plt = Plotter(axes=1, interactive=False)
    plt.show(msh, all_label_points, zoom=1.08, camera={'pos': view1, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot('images/bunny_deformed_subdivided_predictions_view1.png')
    plt.show(msh, all_label_points, zoom=1.08, camera={'pos': view2, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot('images/bunny_deformed_subdivided_predictions_view2.png')


    # Show predictions of perturbed bunny.
    msh = load(f"data/perturbed_bunny_15.obj").c('#ffffff')
    centre_of_mass = msh.points().mean(axis=0)
    preds = np.load("model_outputs/perturb_bunny_15.npy")
    assert len(preds.shape) == 1

    # print(torchmetrics.Accuracy(num_classes=6, average=None, multiclass=True)(torch.IntTensor(preds), torch.IntTensor(labels)))

    all_label_points = get_colored_point_clouds(msh, preds, colormap)
    plt = Plotter(axes=1, interactive=False)
    plt.show(msh, all_label_points, zoom=1.08, camera={'pos': view1, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot('images/perturb_bunny_15_view1.png')
    plt.show(msh, all_label_points, zoom=1.08, camera={'pos': view2, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot('images/perturb_bunny_15_view2.png')


    # Show labels of deformed and decimated bunny.
    msh = load(f"data/{_OBJECT}_rescaled_deformed_decimated_1.obj").c('#ffffff')
    centre_of_mass = msh.points().mean(axis=0)
    labels = np.load("keypoints/bunny/bunny_rescaled_deformed_decimated_1_labels.npy")
    assert len(labels.shape) == 2
    assert np.all(np.sum(labels, axis=1) == 1.)

    labels = np.argmax(labels, axis=1)

    all_label_points = get_colored_point_clouds(msh, labels, colormap)
    plt = Plotter(axes=1, interactive=False)
    plt.show(msh, all_label_points, zoom=1.08, camera={'pos': view1, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot('images/bunny_deformed_decimated_labelled_view1.png')
    plt.show(msh, all_label_points, zoom=1.08, camera={'pos': view2, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot('images/bunny_deformed_decimated_labelled_view2.png')


    # Show predictions of deformed and decimated bunny.
    msh = load(f"data/{_OBJECT}_rescaled_deformed_decimated_1.obj").c('#ffffff')
    centre_of_mass = msh.points().mean(axis=0)
    preds = np.load("model_outputs/bunny_undersampled_deformed.npy")
    assert len(preds.shape) == 1

    all_label_points = get_colored_point_clouds(msh, preds, colormap)
    plt = Plotter(axes=1, interactive=False)
    plt.show(msh, all_label_points, zoom=1.08, camera={'pos': view1, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot('images/bunny_deformed_decimated_predictions_view1.png')
    plt.show(msh, all_label_points, zoom=1.08, camera={'pos': view2, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot('images/bunny_deformed_decimated_predictions_view2.png')


    # Show eigenvectors of original bunny.
    msh = load(f"data/{_OBJECT}_rescaled.obj").c('#ffffff')
    centre_of_mass = msh.points().mean(axis=0)
    eigenvecs = np.load("model_outputs/regular_bunny_100_fourier.npz")['fourier']
    eigenvec = 2
    msh.cmap("viridis", eigenvecs[:, eigenvec])

    plt = Plotter(axes=1, interactive=False)
    plt.show(msh, zoom=1.25, camera={'pos': view1, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot(f'images/bunny_eigenvecs_{eigenvec}_view1.png')
    plt.show(msh, zoom=1.25, camera={'pos': view2, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot(f'images/bunny_eigenvecs_{eigenvec}_view2.png')


    # Show eigenvectors of deformed bunny.
    msh = load(f"data/{_OBJECT}_rescaled_deformed_1.obj").c('#ffffff')
    centre_of_mass = msh.points().mean(axis=0)
    eigenvecs = np.load("model_outputs/bunny_deformed0_100_fourier.npz")['fourier']
    eigenvec = 0
    msh.cmap("viridis", eigenvecs[:, eigenvec])

    plt = Plotter(axes=1, interactive=False)
    plt.show(msh, zoom=1.08, camera={'pos': view1, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot(f'images/bunny_deformed_eigenvecs_{eigenvec}_view1.png')
    plt.show(msh, zoom=1.08, camera={'pos': view2, 'focal_point': centre_of_mass, 'viewup': viewup}).screenshot(f'images/bunny_deformed_eigenvecs_{eigenvec}_view2.png')


if __name__ == "__main__":
    main()

