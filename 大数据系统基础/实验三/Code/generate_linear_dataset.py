import sys
import numpy as np

def generate_and_save_random_dataset(n, m, save_path):
    """
    Generate a random dataset with specified dimensions and save to a text file.

    Parameters:
    n (int): Dimension of x.
    m (int): Number of data points.
    save_path (str): Path to save the generated dataset.
    """
    random_data = np.random.uniform(-2, 2, (m, n + 1))
    np.savetxt(save_path, random_data, fmt='%.2f', delimiter=' ')
    print("Data saved to {}".format(save_path))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <dimension_of_x> <number_of_data_points> <save_path>")
    else:
        n = int(sys.argv[1])  # Dimension of x
        m = int(sys.argv[2])  # Number of data points
        save_path = sys.argv[3]  # Save path for the dataset

        generate_and_save_random_dataset(n, m, save_path)
