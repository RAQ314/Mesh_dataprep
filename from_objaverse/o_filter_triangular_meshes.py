import objaverse
import pandas as pd
from typing import Any, Dict, Hashable
import tqdm
import urllib

import open3d as o3d

import pathlib


def DownloadTriangularMeshesForObjaverse(iMaxNbFaces : int = 500, iDownloadDir : str = "./Objaverse-1.0") :
    #-- Load all uuids and annotations
    all_uids=objaverse.load_uids()
    print(f"Loaded {len(all_uids)} UIDs")
    all_annotations=objaverse.load_annotations(all_uids)

    #--- Create download dir
    download_dir=f"{iDownloadDir}/obj_files_{iMaxNbFaces}"
    pathlib.Path(download_dir).mkdir(parents=True,exist_ok=True)

    #--- Filter uuid refering models with less than maxNumberOfFaces faces
    filteredUids=[]

    for uuid in tqdm.tqdm(all_uids) :
        annotation=all_annotations[uuid]
        if int(annotation["faceCount"])<=iMaxNbFaces :
            filteredUids.append(uuid)

            # if len(filteredUids)>10 :
            #     break

    print(f"Found {len(filteredUids)} objects with less than {iMaxNbFaces} faces")

    #--- Save annotations of filtered objects
    filteredAnnotations=objaverse.load_annotations(filteredUids)
    df=pd.DataFrame.from_dict(filteredAnnotations,orient="index")
    df.to_csv(f"{download_dir}/annotations_less_than_{iMaxNbFaces}_faces.csv")

    #--- Download filtered models
    savedMeshUids=[]

    for filtered_uuid in tqdm.tqdm(filteredUids) :
        localObject=objaverse.load_objects([filtered_uuid])
        localPath=localObject[filtered_uuid]

        mesh=o3d.io.read_triangle_mesh(localPath)
        if len(mesh.triangles)>0 and len(mesh.triangles)<=iMaxNbFaces :
            meshName=pathlib.Path(localPath).stem
            meshToSave=o3d.geometry.TriangleMesh(mesh.vertices, mesh.triangles)
            o3d.io.write_triangle_mesh(f"{download_dir}/{uuid} - {meshName}.obj", meshToSave)
            savedMeshUids.append(filtered_uuid)
        
        pathlib.Path(localPath).unlink()  #--- Delete the downloaded file to save space

    print(f"Saved {len(savedMeshUids)} .obj files with less than {iMaxNbFaces} faces")


if __name__=="__main__" :
    DownloadTriangularMeshesForObjaverse(iMaxNbFaces=500, iDownloadDir="./output/Objaverse-1.0")
    