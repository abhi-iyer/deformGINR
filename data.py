from imports import *
from utils import *

def process_data(n_fourier, path, train=True):
    train_path = os.path.abspath(os.path.join(path, "train"))
    test_path = os.path.abspath(os.path.join(path, "test"))

    train_objs = [each.rstrip(".obj") for each in os.listdir(train_path) if (".obj" in each and "deformed" not in each)]
    test_objs = [each.rstrip(".obj") for each in os.listdir(test_path) if (".obj" in each and "deformed" in each)]
    assert len(train_objs) == 1

    label_file = os.path.join(train_path, train_objs[0] + "_labels.npy")
    assert os.path.exists(label_file)

    # start processing
    process_path = train_path if train else test_path
    process_objs = train_objs if train else test_objs
    processed_files = []

    for obj in process_objs:
        cloud_file = os.path.join(process_path, obj + ".obj")
        output_file = os.path.join(process_path, obj + "_{}_fourier.npz".format(str(n_fourier)))

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
    def __init__(self, dataset_dir, train, n_fourier=100, n_nodes_in_sample=5000):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.train = train
        self.n_fourier = n_fourier
        self.n_nodes_in_sample = n_nodes_in_sample

        self.files = process_data(n_fourier, self.dataset_dir, train)
        self.data = [np.load(f) for _,f in self.files]

    def __getitem__(self, index):
        input_data = torch.from_numpy(self.data[index]["fourier"][:, :self.n_fourier]).float()
        target_data = torch.from_numpy(self.data[index]["target"]).float()

        n_points = input_data.shape[0]
        selected_points = self.get_subsampling_idx(n_points, self.n_nodes_in_sample)

        return input_data[selected_points], target_data[selected_points]

    def get_all_points(self, index):
        if self.train:
            assert index == 0

        return torch.from_numpy(self.data[index]["fourier"][:, :self.n_fourier]).float()

    def __len__(self):
        return len(self.data)

    @property
    def target_dim(self):
        targets = self.data[0]["target"]

        return int(targets.max() - targets.min() + 1)

    @staticmethod
    def get_subsampling_idx(n_points, to_keep):
        if n_points >= to_keep:
            idx = torch.randperm(n_points)[:to_keep]
        else:            
            # Sample some indices more than once
            idx = (
                torch.randperm(n_points * int(np.ceil(to_keep / n_points)))[:to_keep]
                % n_points
            )

        return idx
