import open3d as o3d
import numpy as np
import pymesh


def ComputeEdgeLengthCofficientOfVariation(iTriangularMesh : o3d.geometry.TriangleMesh) -> dict :
    '''Computes the min, max, mean, standard deviation and coefficient of variation of edge lengths of a triangular mesh.'''
    
    heMesh=o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(iTriangularMesh)
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

    return {'mean': edgeLengthMean, 'std': edgeLengthStd, 'smr': edgeLengthsSMR, "min": np.min(edgeLengths), "max": np.max(edgeLengths)}
    return edgeLengthMean, edgeLengthStd, edgeLengthsSMR


def CleanTriMesh(iTriangularMesh : o3d.geometry.TriangleMesh, iDumpInfos = False) -> list :
    '''Cleans a triangular mesh by removing duplicated vertices, duplicated triangles,
    degenerate triangles, non-manifold edges, and unreferenced vertices.
    Slit into connected components
    Returns the list of connected components as triangular meshes.'''

    #--- Fix geometry
    iTriangularMesh.remove_duplicated_vertices()
    iTriangularMesh.remove_duplicated_triangles()
    iTriangularMesh.remove_degenerate_triangles()
    iTriangularMesh.remove_non_manifold_edges()
    iTriangularMesh.remove_unreferenced_vertices()
    
    #--- Split into connected components
    connectedComponents=[]
    
    triangle_clusters, cluster_n_triangles, cluster_area = iTriangularMesh.cluster_connected_triangles()
    nbClusters = len(cluster_n_triangles)
    if nbClusters==1 :
        connectedComponents.append(iTriangularMesh)
    else :
        triangle_clusters = np.asarray(triangle_clusters)
        inputTriangles=np.asarray(iTriangularMesh.triangles)
        inputVertices=np.asarray(iTriangularMesh.vertices)
        
        for ccIndex in range(nbClusters) :
            indices = np.where(triangle_clusters==ccIndex)[0]
            if len(indices)>0 :
                #inputIndices=inputTriangles[indices].reshape((1, -1))
                inputIndices=inputTriangles[indices].flatten()
                ccTriangularMesh = iTriangularMesh.select_by_index(inputIndices)
                #ccTriangularMesh.remove_unreferenced_vertices()
                connectedComponents.append(ccTriangularMesh)

    if iDumpInfos :
        print("Number of connected components : ", len(connectedComponents))
        for ccIndex, ccTriangularMesh in enumerate(connectedComponents) :
            print("  CC #", ccIndex, " : ", len(ccTriangularMesh.vertices), " vertices, ", len(ccTriangularMesh.triangles), " triangles")

    return connectedComponents


def validate_tri_mesh(mesh, iAllowFreeEdges=False) :
  #Watertightness and free edges
  if not mesh.is_watertight() :
    #Open3D defines watertight as : edge + vertex manifold and no self-intersections
    #Here we only check for topology validity, self-intersection is not considered
    if not mesh.is_vertex_manifold() :
       print("Mesh is not vertex manifold")
       return False
    elif not mesh.is_edge_manifold(allow_boundary_edges = iAllowFreeEdges) :
       print("Mesh is not edge manifold")
       return False
    elif mesh.is_self_intersecting() :
       self_intersections = mesh.get_self_intersecting_triangles()
       print(f"Mesh has {len(self_intersections)} self-intersecting triangles : {np.asarray(self_intersections)}")
    elif not iAllowFreeEdges :
       print("Mesh is not watertight")
       return False

  heMesh=o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh)
  if not iAllowFreeEdges :
    nbFreeEdges=0
    for he in heMesh.half_edges :
        if he.twin==-1 : nbFreeEdges+=1
    if nbFreeEdges>0 :
        print(f"Mesh has {nbFreeEdges} free edges")
        return False

  if not mesh.is_edge_manifold() :
    print("Mesh is not manifold")
    return False

  return True


def GetAngleBetweenVectors(iVecA : np.ndarray, iVecB : np.ndarray, iVecNormal : np.ndarray) -> float :
    '''Returns the angle between -180 and 180 in degrees between two vectors around a normal vector.'''
    unitVecANorm=np.linalg.norm(iVecA)
    unitVecBNorm=np.linalg.norm(iVecB)
    unitVecA=iVecA/unitVecANorm if unitVecANorm>0 else iVecA
    unitVecB=iVecB/unitVecBNorm if unitVecBNorm>0 else iVecB
    
    unitVecV=np.cross(iVecNormal, unitVecA)
    unitVecVNorm=np.linalg.norm(unitVecV)
    unitVecV=unitVecV/unitVecVNorm if unitVecVNorm>0 else unitVecV
    
    y=np.dot(unitVecB, unitVecV)
    x=np.dot(unitVecB, unitVecA)
    angleDegrees=np.degrees(np.arctan2(y, x))

    return angleDegrees


def RemoveThinTriangles(iMesh : o3d.geometry.TriangleMesh, iVerbose=False) -> o3d.geometry.TriangleMesh :
    minLengthRatioThreshold=1.05
    coplanarAngleThresholdDegrees=5.0
    
    heMesh=o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(iMesh)
    triangleMasks=np.zeros((len(iMesh.triangles)), dtype=bool)
    
    newTriangles=[]
    
    #--- Perform flip edge on thin triangles
    for currentHE in heMesh.half_edges :
        currentTriangleIndex=currentHE.triangle_index
        otherTriangleIndex=heMesh.half_edges[currentHE.twin].triangle_index if currentHE.twin!=-1 else -1
        if otherTriangleIndex==-1 :
            continue
        if triangleMasks[currentTriangleIndex] or triangleMasks[otherTriangleIndex] :
            #Triangle already marked for removal
            continue
        
        if currentTriangleIndex>otherTriangleIndex :
            v0Id, v1Id = currentHE.vertex_indices[0], currentHE.vertex_indices[1]
            v0=np.array(heMesh.vertices[v0Id])
            v1=np.array(heMesh.vertices[v1Id])
            
            #Direct vertex
            v2Id=heMesh.half_edges[currentHE.next].vertex_indices[1]
            v2=np.array(heMesh.vertices[v2Id])
            
            #Invert vertex
            v3Id=heMesh.half_edges[heMesh.half_edges[currentHE.twin].next].vertex_indices[1]
            v3=np.array(heMesh.vertices[v3Id])
            
            v0v1=np.linalg.norm(v1-v0)
            v0v2=np.linalg.norm(v2-v0)
            v0v3=np.linalg.norm(v3-v0)
            v1v2=np.linalg.norm(v2-v1)
            v1v3=np.linalg.norm(v3-v1)
            directLenghtRatio=(v0v2+v1v2)/v0v1 if v0v1>0 else 0.0
            invertLenghtRatio=(v0v3+v1v3)/v0v1 if v0v1>0 else 0.0
            if directLenghtRatio<minLengthRatioThreshold or invertLenghtRatio<minLengthRatioThreshold :
                #One of the two triangles is too thin
                
                #Check wherther they are near coplanar
                directNormal=np.cross(v1-v0, v2-v0)
                directNorm=np.linalg.norm(directNormal)
                if directNorm>0 :
                    directNormal=directNormal/directNorm
                invertNormal=np.cross(v3-v0, v1-v0)
                invertNorm=np.linalg.norm(invertNormal)
                if invertNorm>0 :
                    invertNormal=invertNormal/invertNorm
                
                angleCosine=np.clip(np.dot(directNormal, invertNormal), -1.0, 1.0)
                angleDegrees=np.degrees(np.arccos(angleCosine))
                if angleDegrees>coplanarAngleThresholdDegrees :
                    continue
                
                #Check whether the quad made by the two triangles is convex so we can flip the edge inside this quad
                #Angles at V0 and V1 should be <180 degrees
                angleV0=GetAngleBetweenVectors(v3-v0, v2-v0, directNormal)
                angleV1=GetAngleBetweenVectors(v2-v1, v3-v1, directNormal)
                if angleV0<0.0 or angleV1<0.0 :
                    #Non-convex quad
                    continue

                #Flip edge
                triangleMasks[currentTriangleIndex]=True
                triangleMasks[otherTriangleIndex]=True
                
                newTriangles.append([v0Id, v3Id, v2Id])
                newTriangles.append([v1Id, v2Id, v3Id])
                    
    if triangleMasks.any() and newTriangles :
        if iVerbose :
            print(f"  Removing {np.sum(triangleMasks)} thin triangles and replacing them with {len(newTriangles)} new triangles")
        iMesh.remove_triangles_by_mask(triangleMasks)
        iMesh.triangles=o3d.utility.Vector3iVector( np.vstack( (np.asarray(iMesh.triangles), np.array(newTriangles)) ) )
        iMesh.remove_unreferenced_vertices()    
        
    return iMesh


def Remesh(iTriangularMesh : o3d.geometry.TriangleMesh, iVerbose=False) -> o3d.geometry.TriangleMesh :
    
    maxNbLoops=3
    minEdgeThresholdRatio=0.2
    maxEdgeThresholdRatio=1.4
    
    for loopIndex in range(maxNbLoops) :
        if iVerbose :
            print(f" Remeshing loop {loopIndex+1}/{maxNbLoops}...")
        
        #Remove thin triangles
        iTriangularMesh=RemoveThinTriangles(iTriangularMesh, iVerbose)
        
        #--- Compute coefficient of variation of edge lengths
        edgeStats = ComputeEdgeLengthCofficientOfVariation(iTriangularMesh)
        
        if iVerbose and (edgeStats["min"]/edgeStats["mean"] > minEdgeThresholdRatio) and (edgeStats["max"]/edgeStats["mean"] < maxEdgeThresholdRatio) :
            print("  Edge lengths are within thresholds, stopping remeshing.")
            break
        
        #Build pymesh from open3d mesh
        pyMesh=pymesh.meshio.form_mesh(np.asarray(iTriangularMesh.vertices), np.asarray(iTriangularMesh.triangles))
        
        minEdgeThresholdValue=minEdgeThresholdRatio*edgeStats["mean"]
        pyMesh, info=pymesh.collapse_short_edges(pyMesh, minEdgeThresholdValue, preserve_feature=True)
                    
        maxEdgeThresholdValue=maxEdgeThresholdRatio*edgeStats["mean"]
        pyMesh, info=pymesh.split_long_edges(pyMesh, maxEdgeThresholdValue)

        #Convert back to open3d mesh
        newTriangularMesh=o3d.geometry.TriangleMesh()
        newTriangularMesh.vertices = o3d.utility.Vector3dVector(pyMesh.vertices.copy())
        newTriangularMesh.triangles = o3d.utility.Vector3iVector(pyMesh.faces.reshape((-1, 3)).copy())
        iTriangularMesh=newTriangularMesh
        
    return iTriangularMesh

    
