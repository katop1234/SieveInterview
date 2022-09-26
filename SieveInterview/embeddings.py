# takes in all the image objects and returns embeddings for them

# using neural networks

# https://www.analyticsvidhya.com/blog/2020/12/an-approach-towards-neural-network-based-image-clustering/
# https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34

# using activeloop tutorial
import glob
data_dir = 'data'

list_imgs = glob.glob(data_dir + "/**/*.jpg")
print(f"There are {len(list_imgs)} images in the dataset {data_dir}")

