# import sys
# import os

# if not __package__:
#     path = os.path.join(os.path.dirname(__file__), os.pardir)
#     sys.path.insert(0, path)

from MLToolbox.neighbors.knn import KNN
def main():
    print('You find the entry_point!')

if __name__ == "__main__":
    main()
    a = KNN()
