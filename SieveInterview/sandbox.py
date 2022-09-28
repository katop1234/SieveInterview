from helpers import get_vector, read_obj, sq_dist_between_two_vectors, get_cnn_model

a = get_vector("frames/frame0/person0.png")[0]
ref_vector = read_obj("refs.center")
whites_vector = read_obj("whites.center")
blues_vector = read_obj("blues.center")

ref_val = sq_dist_between_two_vectors(a, ref_vector)
whites_val = sq_dist_between_two_vectors(a, whites_vector)
blues_val = sq_dist_between_two_vectors(a, blues_vector)

print(len(ref_val))
print(len(a))
