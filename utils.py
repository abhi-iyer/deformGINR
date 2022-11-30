from imports import *


def get_fourier(adj, k=100):
    l = laplacian(adj)

    _, u = sp.linalg.eigsh(l, k=k, which='SM')

    n = l.shape[0]
    u *= np.sqrt(n)

    return u


def laplacian(A):
    return degree_matrix(A) - A


def degree_matrix(A):
    degrees = np.array(A.sum(axis=1)).flatten()

    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)

    return D


def mesh_to_graph(mesh):
    points, edges = pymesh.mesh_to_graph(mesh)

    n = points.shape[0]

    adj = edges_to_adj(edges, n)

    return points, adj


def edges_to_adj(edges, n):
    a = sp.csr_matrix(
        (np.ones(edges.shape[:1]), (edges[:, 0], edges[:, 1])), shape=(n, n)
    )

    a = a + a.T
    a.data[:] = 1.0

    return a


def load_mesh(path, remove_isolated_vertices=True):
    mesh = pymesh.load_mesh(path)

    if remove_isolated_vertices:
        mesh, _ = pymesh.remove_isolated_vertices(mesh)

    return mesh


def degree_power(A, k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        degrees = np.power(np.array(A.sum(1)), k).ravel()

    degrees[np.isinf(degrees)] = 0.0
    
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)

    return D


def normalized_adjacency(A, symmetric=True):
    if symmetric:
        normalized_D = degree_power(A, -0.5)
        return normalized_D.dot(A).dot(normalized_D)
    else:
        normalized_D = degree_power(A, -1.0)
        return normalized_D.dot(A)


def normalized_laplacian(A, symmetric=True):
    if sp.issparse(A):
        I = sp.eye(A.shape[-1], dtype=A.dtype)
    else:
        I = np.eye(A.shape[-1], dtype=A.dtype)

    normalized_adj = normalized_adjacency(A, symmetric=symmetric)
    
    return I - normalized_adj
