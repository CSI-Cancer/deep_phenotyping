import os

import h5py
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
import tqdm

from csi_utils import csi_databases, csi_paths
from csi_images.csi_scans import Scan
from csi_images.csi_tiles import Tile
from csi_images.csi_frames import Frame
from csi_images.csi_events import Event

root_path = "/mnt/deepstore/LBxPheno/train_data/wbc_classifier"
# Read in .csv file with slide IDs
slide_ids = pd.read_csv(os.path.join(root_path, "patients.csv"))["slide_id"].unique()

db = csi_databases.DatabaseHandler("reader", is_production=True)

if os.path.exists(os.path.join(root_path, "all_cells.csv")):
    all_cells = pd.read_csv(os.path.join(root_path, "all_cells.csv"))
else:
    all_cells = []
    for slide_id in slide_ids:
        if slide_id.isnumeric():
            slide_id = "0" + str(slide_id)

        # Query ocular_hitlist for matching frame_id, cell_id, x, and y
        query = f"""
        SELECT slide_id, frame_id, cell_id, x, y, type FROM ocular_hitlist
        WHERE slide_id = %s
        """
        all_cells.append(db.get(query, (slide_id,))["results"])

    # Flatten list of lists
    all_cells = [cell for sublist in all_cells for cell in sublist]

    # Convert to DataFrame
    all_cells = pd.DataFrame(all_cells)
    all_cells.columns = ["slide_id", "frame_id", "cell_id", "x", "y", "type"]
    # Convert frame_id to 1 less for 0-indexing
    all_cells = all_cells.assign(frame_id= all_cells["frame_id"] - 1)
    all_cells = all_cells[[all_cells["type"][i] in ["Rare Cell", "Interesting"] for i in
                           range(len(all_cells))]]
    # Save to .csv
    all_cells.to_csv(os.path.join(root_path, "all_cells.csv"), index=False)

# Gather all scan metadata
all_scans = []
for slide_id in slide_ids:
    if slide_id.isnumeric():
        slide_id = "0" + str(slide_id)

    # Query scan path
    scan_path = csi_databases.query_scan_path(db, slide_id)
    scan = Scan.make_placeholder(slide_id)
    scan.path = scan_path
    scan.channels.append(Scan.Channel("DAPI", 0))
    scan.channels.append(Scan.Channel("AF555", 1))
    scan.channels.append(Scan.Channel("AF647", 2))
    scan.channels.append(Scan.Channel("AF488", 3))
    all_scans.append(scan)

def get_images_for_scan(scan):
    # Get the relevant rows for the current scan
    scan_cells = all_cells[all_cells["slide_id"] == scan.slide_id]
    events = [Event(Tile(scan, frame_id), x, y) for frame_id, x, y in
              zip(scan_cells["frame_id"], scan_cells["x"], scan_cells["y"])]

    # Get the images for the events
    return Event.get_many_crops(events, 75)

with ThreadPoolExecutor() as executor:
    images = list(tqdm.tqdm(executor.map(get_images_for_scan, all_scans), total=len(all_scans)))


# Flatten images from S, N_S, C, H, W to N_total, C, H, W
images = [image for sublist in images for image in sublist ]
images = np.array(images)

# Change from N, C, H, W to N, H, W, C
images = np.transpose(images, (0, 2, 3, 1))

# Create an HDF5 file with the images and features
with h5py.File(os.path.join(root_path, "cells_with_images.hdf5"), "w") as f:
    f.create_dataset("images", data=images)
all_cells.to_hdf(os.path.join(root_path, "cells_with_images.hdf5"), key="features", mode="a")