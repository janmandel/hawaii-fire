import pygrib,os,sys


def count_vertical_levels(file_path):
    # Open the GRIB2 file
    if os.path.isfile(file_path):
        # print('open',file_path)
        grbs = pygrib.open(file_path)
    else:
        print('File',file_path,'does not exist.')
        os.exit(1)

    # Initialize a set to store unique vertical level identifiers
    levels = set()

    # Loop through each message in the GRIB2 file
    for grb in grbs:
        # Check if the message has a level attribute and add it to the set
        if hasattr(grb, 'level'):
            levels.add(grb.level)

    # Close the GRIB2 file
    grbs.close()

    # Return the number of unique vertical levels
    return len(levels)

grib_file_path = sys.argv[1] 
num_levels = count_vertical_levels(grib_file_path)
print(num_levels)

