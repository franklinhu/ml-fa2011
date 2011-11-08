# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011
# Assignment 6

import density_estimation

NUM_ARGS = 1

def uniform_distribution(N):
    return 

def _usage():
    return "Usage: python density_mixture.py data_file"

if __name__ == "__main__":
    if len(sys.argv) < NUM_ARGS + 1:
        print _usage()
        sys.exit(1)

    data_file = sys.argv[1]
    handle = open(data_file, 'r')
    handle.readline()
    csv_file = csv.reader(handle)
    data = []

    for line in csv_file:
        l = tuple(map(lambda x: float(x), line[0:2]))
        data.append(l)

    laplacian_weight = 0.5


