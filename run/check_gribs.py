import pygrib
import os
from datetime import datetime, timedelta

def count_vertical_levels(grib_file):
    """Counts the unique vertical levels in a GRIB2 file."""
    if not os.path.isfile(grib_file):
        return None  # File does not exist

    grbs = pygrib.open(grib_file)
    levels = {grb.level for grb in grbs if hasattr(grb, 'level')}
    grbs.close()

    return len(levels)

def check_files_and_levels(start_date, end_date, template):
    """Checks for file gaps and changes in number of levels from start_date to end_date."""
    current_date = start_date
    last_level_count = None
    gap_dates = []
    level_changes = []

    while current_date <= end_date:
        # Format the date components for the file path
        year = current_date.strftime("%Y")
        month = current_date.strftime("%m")
        day = current_date.strftime("%d")
        hour = "0000"  # Modify if the hours change in the file naming convention

        # Construct file path
        file_path = template.format(year=year, month=month, day=day, hour=hour)

        # Check if file exists and count levels
        level_count = count_vertical_levels(file_path)

        if level_count is None:
            # File is missing
            gap_dates.append(current_date)
        else:
            # Check for changes in level count
            if last_level_count is not None and level_count != last_level_count:
                level_changes.append((current_date, last_level_count, level_count))
            last_level_count = level_count

        # Move to the next day
        current_date += timedelta(days=1)

    # Output results
    print("Missing files (gaps):")
    for date in gap_dates:
        print(date.strftime("%Y-%m-%d"))

    print("\nLevel changes:")
    for date, old_count, new_count in level_changes:
        print(f"{date.strftime('%Y-%m-%d')}: {old_count} -> {new_count}")

# Define the date range and file path template
start_date = datetime(2011, 1, 1)  # Start date of the data
end_date = datetime.now()  # Until the current date
file_template = "wrfxpy/ingest/GFSA/{year}{month}/{year}{month}{day}/gfsanl_4_{year}{month}{day}_{hour}_000.grb2"

# Run the check
check_files_and_levels(start_date, end_date, file_template)

