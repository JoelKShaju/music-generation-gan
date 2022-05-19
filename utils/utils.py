import os
def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g.TRABAFJ128F42AF24E -> A/B/A/TRABAFJ128F42AF24E"""
    
    return os.path.join(msd_id[2],msd_id[3], msd_id[4], msd_id)