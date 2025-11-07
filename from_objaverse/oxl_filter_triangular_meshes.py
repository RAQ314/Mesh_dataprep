import objaverse
import objaverse.xl as oxl
import pandas as pd
from typing import Any, Dict, Hashable
import tqdm
import urllib

import open3d as o3d

import pathlib


def DownloadTriangularMeshesForObjaverseXL(iMaxNbFaces : int = 500, iDownloadDir : str = "./Objaverse-1.0") :
    #----- Select all .obj files
    annotations_df=oxl.get_annotations(download_dir="./output/.objaversexl_annotations")

    selected_annotations_df=annotations_df[
        (annotations_df["fileType"]=="obj")]
    print(f"Found {len(selected_annotations_df)} .obj files")

    selected_annotations_df.head(50).to_csv("./output/objaversexl_obj_files.csv", index=False)

    #--- Create download dir 
    download_dir=f"{iDownloadDir}/obj_files_{iMaxNbFaces}"
    pathlib.Path(download_dir).mkdir(parents=True,exist_ok=True)

    validObjects=[]

    #----- Download all .obj files with less than 500 faces
    
    # #Callback of downloaded model
    # def handle_found_object(local_path: str,
    #                         file_identifier: str,
    #                         sha256: str,
    #                         metadata: Dict[Hashable, Any]) -> None:
        
    #     #Open local mesh
    #     mesh=o3d.io.read_triangle_mesh(local_path)
    #     if len(mesh.triangles)<=iMaxNbFaces :
    #         pass
    #         # validObjects.append({
    #         #     "local_path": local_path,
    #         #     "file_identifier": file_identifier,
    #         #     "sha256": sha256,
    #         #     "num_faces": len(mesh.triangles),
    #         #     **metadata
    #         # })
    #     else :
    #         pathlib.Path(local_path).unlink()

    # #Download models
    # oxl.download_objects(objects=selected_annotations_df,
    #                      handle_found_object=handle_found_object,
    #                      download_dir=f"./output/objaverse_obj_files_{iMaxNbFaces}",
    #                      processes=1)

    for rowIndex in tqdm.tqdm(range(len(selected_annotations_df))):
        try :
            local_path=oxl.download_objects(selected_annotations_df.iloc[[rowIndex]], download_dir=download_dir, processes=1)
        except Exception as e:
            print(f"Error downloading object at index {rowIndex}: {e}")
            continue

        #Open local mesh
        mesh=o3d.io.read_triangle_mesh(local_path)
        
        if len(mesh.triangles)>0 and len(mesh.triangles)<=iMaxNbFaces :
            #pass
            validObjects.append({
                    "local_path": local_path,
                    "file_identifier": row.fileIdentifier,
                    "sha256": row.sha256,
                    "num_faces": len(mesh.triangles),
                    **row.metadata
                })
        else :
            pathlib.Path(local_path).unlink()


    print(f"Found {len(validObjects)} .obj files with less than {iMaxNbFaces} faces.")
    validObjects_df=pd.DataFrame(validObjects)
    validObjects_df.to_csv(f"{iDownloadDir}/obj_files_less_than_{iMaxNbFaces}_faces.csv", index=False)


if __name__=="__main__" :
    DownloadTriangularMeshesForObjaverseXL(iMaxNbFaces=500, iDownloadDir="./output/ObjaverseXL")
