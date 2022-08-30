# model to detect all person frames

# have a helper function that can pull up the person object's frame for a target
# frame to make manual classification easier.

# manually parse through some of the frames for each desired category and get
# their embedding. store these somewhere as reference

# go through each person objects recognized and calculate the sum euclidean
# distance to each of the desired categories, and pick the top n of each to
# classify as being the that type. also have a way to store the objects in a
# json for each frame, should be given to you in the setup for the models.2
