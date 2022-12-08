from imports import *
from utils import *

def process_data(n_fourier, path, train=True):
    train_path = os.path.abspath(os.path.join(path, "train"))
    test_path = os.path.abspath(os.path.join(path, "test"))

    train_objs = [each.rstrip(".obj") for each in os.listdir(train_path) if ".obj" in each]
    test_objs = [each.rstrip(".obj") for each in os.listdir(test_path) if ".obj" in each]
    assert len(train_objs) == 1

    # start processing
    process_path = train_path if train else test_path
    process_objs = train_objs if train else test_objs
    processed_files = []

    for obj in process_objs:
        cloud_file = os.path.join(process_path, obj + ".obj")
        label_file = os.path.join(process_path, obj + "_labels.npy")
        output_file = os.path.join(process_path, obj + "_{}_fourier.npz".format(str(n_fourier)))

        assert os.path.exists(cloud_file)
        assert os.path.exists(label_file)

        if not os.path.exists(output_file):
            mesh = load_mesh(cloud_file)
            labels = np.load(label_file).argmax(axis=1)

            points, adj = mesh_to_graph(mesh)

            u = get_fourier(adj, n_fourier)

            np.savez(
                output_file,
                points=points,
                fourier=u,
                target=labels,
                faces=mesh.faces,
            )

        processed_files.append((obj, output_file))

    return processed_files


class GraphDataset(Dataset):
    def __init__(self, dataset_dir, train, n_fourier=100):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.train = train
        self.n_fourier = n_fourier

        self.obj_to_output_file = process_data(n_fourier, self.dataset_dir, train)
        self.data = [np.load(f) for _,f in self.obj_to_output_file]
        
        self.all_points = torch.from_numpy(np.concatenate([each["fourier"] for each in self.data], axis=0)).float()
        self.all_labels = torch.from_numpy(np.concatenate([each["target"] for each in self.data], axis=0)).long()

    def __getitem__(self, index):
        return self.all_points[index, :self.n_fourier], self.all_labels[index]

    def get_object_points(self, index):
        assert 0 <= index <= (len(self.data)-1)

        points = torch.from_numpy(self.data[index]["fourier"][:, :self.n_fourier]).float()
        labels = torch.from_numpy(self.data[index]["target"]).long()

        return points, labels

    def __len__(self):
        return len(self.all_points)

    @property
    def num_objects(self):
        return len(self.data)

    @property 
    def label_weight(self):
        return 1/F.one_hot(self.all_labels).sum(dim=0)

    @property
    def target_dim(self):
        return int(self.all_labels.max() - self.all_labels.min() + 1)
