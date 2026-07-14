import os
import numpy as np

def combine_text_files(input_dir, output_file):
    coords_all = []
    for filename in sorted(os.listdir(input_dir)):
        if filename.startswith('vortex_position') and filename.endswith('.txt'):
            date = filename.split('_')[2]
            time = filename.split('_')[-1][:-4]
            tstring = date + '_' + time
            file_path = os.path.join(input_dir, filename)
            vortex = np.loadtxt(file_path, delimiter=',')
            vlon = float(vortex[0])
            vlat = float(vortex[1])
            coords_all.append((tstring, vlat, vlon))  # Append as tuple
    # Convert to a regular object array
    coords_array = np.array(coords_all, dtype=object)
    header = "Time [YYYYMMDD_HHMMSS UTC], Vortex latitude, Vortex longitude"
    np.savetxt(output_file, coords_array, header=header, delimiter=", ", fmt=["%s", "%.6f", "%.6f"])

if __name__ == "__main__":
    input_directory = "/Users/robbyfrost/Desktop/20250518_figs/tornado1/"
    output_filename = "tornado1_vortex_positions.txt"
    combine_text_files(input_directory, output_filename)