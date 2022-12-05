from imports import *
from utils import *

def process_data(path="./data", train=True):
    path = os.path.abspath(os.path.join(path, "train" if train else "test"))

    objects = [each.rstrip(".stl") for each in os.listdir(path) if ".stl" in each]

    npz_files = []

    for o in objects:
        cloud_file = os.path.join(path, o + ".stl")
        label_file = os.path.join(path, o + "_labels.npy")
        output_file = os.path.join(path, o + ".npz")

        assert os.path.exists(label_file)

        if not os.path.exists(output_file):
            mesh = load_mesh(cloud_file)
            labels = np.load(label_file).argmax(axis=1)

            points, adj = mesh_to_graph(mesh)

            u = get_fourier(adj)

            np.savez(
                output_file,
                points=points,
                fourier=u,
                target=labels, 
                faces=mesh.faces,
            )

        npz_files.append(output_file)

    return sorted(npz_files)


class GraphDataset(Dataset):
    def __init__(self, dataset_dir, train, n_fourier=100, n_nodes_in_sample=5000):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.n_fourier = n_fourier
        self.n_nodes_in_sample = n_nodes_in_sample

        self.filenames = process_data(self.dataset_dir, train)
        self.npzs = [np.load(f) for f in self.filenames]


    def __getitem__(self, index):
        data = {}

        input_data = torch.from_numpy(self.npzs[index]["fourier"][:, :self.n_fourier]).float()
        target_data = torch.from_numpy(self.npzs[index]["target"]).float()

        n_points = input_data.shape[0]
        points_idx = self.get_subsampling_idx(n_points, self.n_nodes_in_sample)

        data["inputs"] = input_data[points_idx]
        data["targets"] = target_data[points_idx]
        data["index"] = index    

        return data

    def get_full(self, index):
        data = {}

        input_data = torch.from_numpy(self.npzs[index]["fourier"][:, :self.n_fourier]).float()
        target_data = torch.from_numpy(self.npzs[index]["target"]).float()

        return input_data, target_data, index

    def __len__(self):
        return len(self.filenames)

    @property
    def target_dim(self):
        targets = self[0]["targets"]

        return int((targets.max() - targets.min() + 1).item())

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
