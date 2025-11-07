import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

import numpy as np
import open3d as o3d
from pathlib import Path
# from fns import dequantize_verts_tensor, create_io_sequence, quantize_verts, prepare_halfedge_mesh, normalize_vertices_scale
# import torch
# from einops import pack
# from utils.utils import CleanGeneratedTriangles
from utils.mesh_validation import validate_tri_mesh, CleanTriMesh, Remesh
import tqdm
import glob
import pandas as pd


#---- Parameters
ENABLE_REMESH=True
VERBOSE=False

inputDirectory="./data/mesh_500/obj/train"
outputDirectory="./output/CleanedMeshes_mesh500_train"
inputDirectory="./data/mesh_500/obj/val"
outputDirectory="./output/CleanedMeshes_mesh500_val"
inputDirectory="./data/Thingi10K_Test"
outputDirectory="./output/CleanedMeshes_Thingi10K_Test"


#Count number of files to process
nbFilesToProcess=0
for file in os.listdir(inputDirectory) :
  if file.endswith(".obj") :
    nbFilesToProcess+=1

#Process all files
print(f"Processing {nbFilesToProcess} files in {inputDirectory}...")

Path(outputDirectory).mkdir(parents=True, exist_ok=True)


df=pd.DataFrame({ "name":[], "nb faces":[], "nb vertices": [] , "edge length mean": [], "edge length std": [], "edge length smr" : [] })
def AddMeshStats(iMeshName : str, iMesh : o3d.geometry.TriangleMesh) :
    df_line=[]
    df_line.append(iMeshName)
    df_line.append(len(iMesh.triangles))
    df_line.append(len(iMesh.vertices))
    
    heMesh=o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(iMesh)
    edgeLengths=[]
    for currentHE in heMesh.half_edges :
        currentTriangleIndex=currentHE.triangle_index
        otherTriangleIndex=heMesh.half_edges[currentHE.twin].triangle_index if currentHE.twin!=-1 else -1
        if currentTriangleIndex>otherTriangleIndex :
            v0Id, v1Id = currentHE.vertex_indices[0], currentHE.vertex_indices[1]
            v0=np.array(heMesh.vertices[v0Id])
            v1=np.array(heMesh.vertices[v1Id])
            edgeLengths.append(np.linalg.norm(v1-v0))
    
    edgeLengthMean=np.mean(edgeLengths)
    edgeLengthStd=np.std(edgeLengths)
    edgeLengthsSMR=edgeLengthStd/edgeLengthMean if edgeLengthMean>0 else 0.0
    df_line.append(edgeLengthMean)
    df_line.append(edgeLengthStd)
    df_line.append(edgeLengthsSMR)
    df.loc[len(df)] = df_line


with open(outputDirectory+"/CleanMeshes.log", "w") as resultFile :
    for filename in tqdm.tqdm(glob.iglob(inputDirectory + '**/**', recursive=True), total=nbFilesToProcess) :
        if filename.endswith(".obj") :
            print(f"--> Processing {filename}")
            
            try :
                mesh = o3d.io.read_triangle_mesh(filename)
                connexComponents=CleanTriMesh(mesh)
                basename=os.path.basename(filename)
                
                if len(connexComponents)==1 :
                    if not validate_tri_mesh(connexComponents[0], iAllowFreeEdges=True) :
                        resultFile.write(f"Mesh {filename} is not valid after cleaning\n")
                    else :
                        if ENABLE_REMESH :
                            #Remesh
                            tempMesh=Remesh(connexComponents[0], VERBOSE)
                            if validate_tri_mesh(tempMesh, iAllowFreeEdges=True) :
                                o3d.io.write_triangle_mesh(outputDirectory+"/"+basename, tempMesh)
                                AddMeshStats(basename, tempMesh)
                        else :
                            o3d.io.write_triangle_mesh(outputDirectory+"/"+basename, connexComponents[0])
                            AddMeshStats(basename, connexComponents[0])
                elif len(connexComponents)>1 :
                    for i, cc in enumerate(connexComponents) :
                        name, ext = os.path.splitext(basename)
                        if not validate_tri_mesh(cc, iAllowFreeEdges=True) :
                            resultFile.write(f"Mesh {name}_cc{i}{ext} is not valid after cleaning\n")
                        else :
                            if ENABLE_REMESH :
                                #Remesh
                                o3d.io.write_triangle_mesh(outputDirectory+"/"+"INPUT"+name+f"_cc{i}"+ext, cc)
                                tempMesh=Remesh(cc, VERBOSE)
                                if validate_tri_mesh(tempMesh, iAllowFreeEdges=True) :
                                    o3d.io.write_triangle_mesh(outputDirectory+"/"+name+f"_cc{i}"+ext, tempMesh)
                                    AddMeshStats(basename, tempMesh)
                                    
                                    # break  #--- DEBUG ONLY
                            else :
                                o3d.io.write_triangle_mesh(outputDirectory+"/"+name+f"_cc{i}"+ext, cc)
                                AddMeshStats(name+f"_cc{i}"+ext, cc)
            except Exception as e :
                resultFile.write(f"Error while processing {filename} : {e}\n")
                
            # if ENABLE_REMESH :
            #     break  #--- DEBUG ONLY

#Save meshes stats
df.to_csv(outputDirectory+"/MeshesStats.csv", index=False)

