import requests
from requests.adapters import HTTPAdapter, Retry
import json

class PoseDatabase:

    def __init__(self, hostname="localhost"):
        """ Constructor """
        self._hostname = hostname

    def get_pose_from_db(self, pose_name):
        # Attempt to retrieve the specified pose from the database
    
        pose = []
        api_url = "http://" + self._hostname + ":8000/api/poses"
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        response = requests.get(api_url, params={'search': pose_name})
        pose_data = response.json()
        if pose_data['status'] == 'success':
            if pose_data['results'] > 0:
                if pose_data['results'] == 1:
                    pose = pose_data['poses'][0]['data']
                else:
                    print("Got multiple matches with the name: "+ pose_name + "check that the pose is unique or use the pose ID")
                    return []
            else:
                print("Did not find any pose with the name: "+ pose_name +" in the database.")
                return []
        else:
            print("Failed to get pose from database, check the IP of the pose database!")
            return []
        
        return pose
