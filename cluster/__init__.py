"""K-means helpers; scikit-learn is required only when loading a cluster model."""
import torch


def get_cluster_model(ckpt_path):
    from sklearn.cluster import KMeans
    checkpoint = torch.load(ckpt_path)
    kmeans_dict = {}
    for spk, ckpt in checkpoint.items():
        km = KMeans(ckpt["n_features_in_"])
        km.__dict__["n_features_in_"] = ckpt["n_features_in_"]
        km.__dict__["_n_threads"] = ckpt["_n_threads"]
        km.__dict__["cluster_centers_"] = ckpt["cluster_centers_"]
        kmeans_dict[spk] = km
    return kmeans_dict


def get_cluster_result(model, x, speaker):
    """Return cluster labels for features shaped [frames, channels]."""
    return model[speaker].predict(x)


def get_cluster_center_result(model, x, speaker):
    """Return the nearest center for each input feature frame."""
    return model[speaker].cluster_centers_[get_cluster_result(model, x, speaker)]


def get_center(model, x, speaker):
    return model[speaker].cluster_centers_[x]
