#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Imports
import os
import glob
import time
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
import trimesh
from trimesh.curvature import (
    discrete_gaussian_curvature_measure,
    discrete_mean_curvature_measure,
    sphere_ball_intersection,
)
import pyvista as pv

# pv.set_jupyter_backend("server")
# Set interactive environment
# pv.set_jupyter_backend('trame')
import pymeshfix as mf
from pymeshfix import MeshFix
from pymeshfix._meshfix import PyTMesh
from pymeshfix.examples import planar_mesh

# Import meshlib
from meshlib import mrmeshpy as mm
from meshlib import mrmeshnumpy as mn

# Import vedo
import vedo
from vedo import *

# Import open 3D
import open3d as o3d
from open3d import *

from datetime import datetime


# In[8]:


# Functions

##################################################################
# Scene Extraction ###############################################
##################################################################


# Tissue Classification
def tissue_scene(file_path):
    # Load File
    nifti_file = nib.load(file_path)  # load file from the path
    segmentation_array = (
        nifti_file.get_fdata()
    )  # convert nifti segmentation file into numpy array

    structure = [24, 25]  # Mask Labels
    tissue_scene = np.where(
        (segmentation_array != 0)  # remove background
        & np.isin(
            segmentation_array, structure
        ),  # flag structure of interest
        1,
        0,
    )  # assign values

    return tissue_scene, nifti_file


##################################################################
# Mesh Processing ################################################
##################################################################


def seg_to_mesh(tissue_class_path):  # convert segmentation to mesh

    # Extract Masks and Nifti Files
    tissue_mask, tissue_nifti = tissue_scene(tissue_class_path)
    # Atlas and Tissue Classification Intersection
    scene = tissue_mask

    # Use Vedo Surfnets to Create Basic Isosurface as Reference and Isosurface Discrete for Further Processing

    # Extract volume
    vol = Volume(
        scene, spacing=tissue_nifti.header.get_zooms()
    )  # Extract volume and correct strectching along all axis

    # Create mesh from mask
    iso = vol.isosurface()  # non-discreate
    iso_d = vol.isosurface_discrete(
        [
            0,
        ]
    )  # discreate

    # Return Vedo Meshes
    return iso, iso_d


# Correct Mesh
def correct_mesh(iso):
    # Extract vertices and faces
    vertices = iso.vertices
    faces = iso.cells
    # Clean vertices and faces
    vertices, faces = mf.clean_from_arrays(vertices, faces)  #### MAIN DIFFERENCE ###

    # Load PyTMesh
    mfix = PyTMesh(False)  # False removes extra verbose output
    # Create array
    mfix.load_array(vertices, faces)

    # Fix mesh
    # Fills all the holes having at at most 'nbe' boundary edges. If
    # 'refine' is true, adds inner vertices to reproduce the sampling
    # density of the surroundings. Returns number of holes patched.  If
    # 'nbe' is 0 (default), all the holes are patched.
    mfix.fill_small_boundaries(nbe=0, refine=True)

    # Connect Closest Components
    mfix.join_closest_components()

    # Remove Islands
    mfix.remove_smallest_components()

    # Return vertices and faces
    vert, faces = mfix.return_arrays()
    triangles = np.empty((faces.shape[0], 4), dtype=faces.dtype)
    triangles[:, -3:] = faces
    triangles[:, 0] = 3
    # Create mesh
    mesh = pv.PolyData(vert, triangles)
    # Return PyVista Mesh
    return mesh


# Hole Fill and Floating Island Removal
def mesh_fix(iso):

    # Extract Vertices and Faces
    vertices = iso.vertices
    faces = np.array(iso.cells)

    # Load PyTMesh
    mfix = PyTMesh(False)  # False removes extra verbose output
    # Create array
    mfix.load_array(vertices, faces)

    # Fix mesh
    # Fills all the holes having at at most 'nbe' boundary edges. If
    # 'refine' is true, adds inner vertices to reproduce the sampling
    # density of the surroundings. Returns number of holes patched.  If
    # 'nbe' is 0 (default), all the holes are patched.
    mfix.fill_small_boundaries(nbe=0, refine=True)

    # Connect Closest Components
    mfix.join_closest_components()

    # Remove Islands
    mfix.remove_smallest_components()

    # Return vertices and faces
    vertices, faces = mfix.return_arrays()

    # Get Trieangles
    triangles = np.empty((faces.shape[0], 4), dtype=faces.dtype)
    triangles[:, -3:] = faces
    triangles[:, 0] = 3

    # Convert to PyVista Mesh
    pv_mesh = pv.PolyData(vertices, triangles)

    # Return PyVista Mesh
    return pv_mesh


# Taubin Smoothing
def taubin_smooth(pv_mesh):
    # Smooth Mesh
    pv_mesh = pv_mesh.smooth_taubin()

    # Return PyVista Mesh
    return pv_mesh


# Subdivide and Decimate to Specific Size
def subd_decim(pv_mesh):
    # Filter
    subfilter = "loop"  # butterfly linear loop
    # Subdivide
    pv_mesh = pv_mesh.subdivide(2, subfilter=subfilter)
    # Reduce Complexity to Specified Target
    pv_mesh = pv_mesh.decimate(
        1 - (1 / pv_mesh.n_points * 18000), volume_preservation=True
    )

    # Return PyVista Mesh
    return pv_mesh


# Convert Vedo to Pyvista
def vedo_to_pv_mesh(vedo_mesh):
    # Convert to Trimesh
    tri_mesh = vedo.vedo2trimesh(vedo_mesh)
    # Convert to Pyvista
    pv_mesh = pv.wrap(tri_mesh)
    # Return Mesh
    return pv_mesh


# Convert Pyvista to Vedo
def pv_to_vedo(pv_mesh):
    # Extract Vertices
    pv_vertices = pv_mesh.points
    # Extract Faces
    pv_faces = pv_mesh.faces
    # Create Vedo Mesh
    vedo_mesh = vedo.Mesh([pv_vertices, pv_faces])
    # Return Mesh
    return vedo_mesh


# Convert Pyvista to Trimesh
def pv_to_tri(pv_mesh):
    # Convert to Vedo
    vedo_mesh = pv_to_vedo(pv_mesh)
    # Convert to Trimesh
    tri_mesh = vedo.vedo2trimesh(vedo_mesh)
    # Return Mesh
    return tri_mesh

# Function to Subdivide if Mesh is very Small Followed by Decimation to Standard Size  and Smoothing
def process_mesh(pv_mesh):
    ### Processing Parameters ###
    itr = 5
    lamb = 0.5
    rdecim = 0.5
    ndecim = 12000
    # Convert to Vedo Mesh
    iso = pv_to_vedo(pv_mesh)
    # ### Subdivide and Decimate Small Meshes ###
    if iso.vertices.shape[0] < ndecim:  # check size
        # Extract Vertices
        vedo_vertices = iso.vertices
        # Extract Faces
        vedo_faces = iso.cells
        # Subdivide Mesh
        tri_vertices, tri_faces = trimesh.remesh.subdivide_loop(
            vedo_vertices, np.array(vedo_faces).astype(int), iterations=1
        )
        # Tri Mesh
        tri_iso = trimesh.Trimesh(tri_vertices, tri_faces)
        # Convert to Vedo Mesh
        iso = vedo.trimesh2vedo(tri_iso)
    ### Smooth Mesh ###
    # Correct Mesh
    pv_mesh = pv.wrap(correct_mesh(iso))
    # Convert to Tri Mesh
    tri_mesh = pv_to_tri(pv_mesh)
    # Smooth Mesh
    trimesh.smoothing.filter_laplacian(
        tri_mesh,
        lamb=lamb,
        iterations=itr,
    )
    # Convert to Vedo Mesh
    iso = vedo.trimesh2vedo(tri_mesh)
    # Decimate
    iso = iso.decimate(n=ndecim, regularization=rdecim)  # decimate

    ### Return Vedo Mesh ###
    return iso


def gauss_curv(tri_mesh, radii_range):
    # Compute Gaussian Curvature
    gauss = np.array(
        [
            # Gaussian Curvature
            discrete_gaussian_curvature_measure(tri_mesh, tri_mesh.vertices, r)
            # Mean Curvature
            # discrete_mean_curvature_measure(tri_mesh, tri_mesh.vertices, r)
            # / sphere_ball_intersection(1, r)
            for r in radii_range
        ]
    ).mean(axis=1)

    return gauss


# Averaged Local Features
def local_shape_measurments(
    pv_mesh, r=1.0
):  # set search radius [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05]
    # Load the point cloud and normals
    point_cloud = np.asarray(pv_mesh.points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # # Remove Outliers ### EXPERIMENTAL ###########################
    #     # Downsample the Point Cloud with a Voxel of 0.01
    #     voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.01)
    #     # Statistical Oulier Removal
    #     pcd, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=100,
    #                                                         std_ratio=2.0)
    # ### END OF EXPERIMENTA #######################################

    pcd.compute_convex_hull()
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(1)

    mesh_verts = np.asarray(pcd.points)
    mesh_norms = np.asarray(pcd.normals)

    # #  Transform the point cloud to the a PointCloud() object used by open3d
    pcd.points = utility.Vector3dVector(mesh_verts)
    pcd.normals = utility.Vector3dVector(mesh_norms)

    # Calculate the KDTree
    pcd_tree = geometry.KDTreeFlann(pcd)

    allVerts = np.ones([len(mesh_verts), 1])

    # how many scales will be used
    numScales = 1

    # Place holder for set of features
    feature_set = []

    # Data Container
    data_dict = {}

    i = 0

    #  go through all the points
    while i < len(mesh_verts):

        pointScaleFeatures = []

        for j in [r]:

            # Constant for Stability
            constant = 0.0
            #  for each radius area scale find neighbours
            [k_small, idx_small, distances_small] = pcd_tree.search_radius_vector_3d(
                pcd.points[i], j
            )  # radius search set radius

            currNdx = np.array(idx_small)
            #  if there are less than 2 neightbours just add 0s
            if len(currNdx) <= 2:
                linearity = 0 + constant
                planarity = 0 + constant
                sphericity = 0 + constant
                omnivariance = 0 + constant
                anisotropy = 0 + constant
                eigenentropy = 0 + constant
                sumOFEigs = 0 + constant
                changeOfCurvature = 0 + constant
                farthestDist = distances_small[len(distances_small) - 1] / j
                pointDensity = k_small / j
                heightStd = 0 + constant
                heightMax = 0 + constant

            #  if there are enouigh neighbours then continue with computation
            else:

                #  get heighbourhood points and normals
                nearestNeighbors_normals = mesh_norms[currNdx]
                nearestNeighbors_verts = mesh_verts[currNdx]

                #  calculate the covariance matrix, and eigenvalues
                cov_mat = np.cov(
                    [
                        nearestNeighbors_verts[:, 0],
                        nearestNeighbors_verts[:, 1],
                        nearestNeighbors_verts[:, 2],
                    ]
                )
                eig_val_cov, eig_vec_cov = np.linalg.eigh(cov_mat)
                idx = eig_val_cov.argsort()[::-1]
                eig_val_cov = eig_val_cov[idx]

                #  calculate the first 12 features derived from shape measurements
                linearity = (eig_val_cov[0] - eig_val_cov[1]) / eig_val_cov[0]
                planarity = (eig_val_cov[1] - eig_val_cov[2]) / eig_val_cov[0]
                sphericity = eig_val_cov[2] / eig_val_cov[0]
                omnivariance = (eig_val_cov[0] * eig_val_cov[1] * eig_val_cov[2]) ** (
                    1.0 / 3.0
                )
                anisotropy = (eig_val_cov[0] - eig_val_cov[2]) / eig_val_cov[0]
                eigenentropy = -(
                    (eig_val_cov[0] * np.log(eig_val_cov[0]))
                    + (eig_val_cov[1] * np.log(eig_val_cov[1]))
                    + (eig_val_cov[2] * np.log(eig_val_cov[2]))
                )
                sumOFEigs = eig_val_cov[0] + eig_val_cov[1] + eig_val_cov[2]
                changeOfCurvature = eig_val_cov[2] / (
                    eig_val_cov[0] + eig_val_cov[1] + eig_val_cov[2]
                )

                farthestDist = distances_small[len(distances_small) - 1] / j
                pointDensity = k_small / j
                heightMax = (
                    np.abs(
                        np.dot(
                            nearestNeighbors_normals.mean(axis=0),
                            nearestNeighbors_verts.T,
                        )
                    ).max()
                    - np.abs(
                        np.dot(
                            nearestNeighbors_normals.mean(axis=0),
                            nearestNeighbors_verts.T,
                        )
                    ).min()
                )
                heightStd = np.abs(
                    np.dot(
                        nearestNeighbors_normals.mean(axis=0), nearestNeighbors_verts.T
                    )
                ).std()

            #  Added to the other features and a feature vector is created
            pointScaleFeatures.extend(
                [
                    linearity,
                    planarity,
                    sphericity,
                    omnivariance,
                    anisotropy,
                    eigenentropy,
                    sumOFEigs,
                    changeOfCurvature,
                    farthestDist,
                    pointDensity,
                    heightMax,
                    heightStd,
                ]
            )

        #  The feature vector for each point is checked for NaN values and then concatenated
        where_are_NaNs = np.isnan(pointScaleFeatures)
        pointScaleFeatures = np.array(pointScaleFeatures)
        pointScaleFeatures[where_are_NaNs] = 0
        pointScaleFeatures = pointScaleFeatures.tolist()
        feature_set.append(pointScaleFeatures)

        i += 1

    # DataFrame of averaged features
    col_names = [
        "linearity",
        "planarity",
        "sphericity",
        "omnivariance",
        "anisotropy",
        "eigenentropy",
        "sumOFEigs",
        "changeOfCurvature",
        "farthestDist",
        "pointDensity",
        "heightStd",
        "heightMax",
    ]
    feature_df = pd.DataFrame(
        feature_set, columns=col_names
    )  # TO GET ENTIRE DATA FOR PLOTTTING

    # Get  values
    final_df = feature_df.mean()

    return final_df


# In[9]:


# Date
date = datetime.today().strftime("%d-%m-%Y")
print(date)

# Source and Destination Paths
src_dir_path = "/Users/apenkaus/Documents/SSM2"  # source path
dst_dir_path = f"_{date}" # destination path
feature_df_path = f"feature-df_{date}.csv" # path for feature data csv
print(dst_dir_path)
# Subject List
sub_list = sorted([sub for sub in os.listdir(src_dir_path) if "sub" in sub])
print (f'Subject Count: {len(sub_list)}')


# In[5]:


# Perturbation Setup

# Radii Range
# radii_range = np.round(np.arange(0.1,10.1,0.1),2) # large
radii_range = np.round(np.arange(0.5, 1.6, 0.1), 2)  # small


# Create Data Container
# Radii Column Names
radii_cols = radii_range.astype(str)  # measurment columns
# Global and Local Feature Tags
ft_cols = [
    "SA",
    "V",
    "SAVR",
    "IMC",
    "As",
    "AsE",
    "CHV",
    "CHSA",
    "CHSAVR",
    "CHSAR",
    "CHVR",
]

cols = ["SUB_ID", "SES", "TAG"]  # subject & session & mesh tag
cols.extend(radii_cols)  # combined with radii
cols.extend(ft_cols)  # combined with global and local


# Main DataFrame
data_frame = pd.DataFrame(columns=cols)  # fresh START
# data_frame = pd.read_csv('.csv', index_col=0) # read .csv file to continue in case of ERROR

# Error Container
error_dict = {}

for sub in sub_list[:1]:
    # print(f'Subject: {sub}')
    sub_path = os.path.join(src_dir_path, sub)
    session_list = sorted([ses for ses in os.listdir(sub_path) if "ses" in ses])
    print(sub_path)
    print(session_list)
    for ses in session_list:
        if (
            len(data_frame[(data_frame["SUB_ID"] == sub) & (data_frame["SES"] == ses)])
            < 1
        ):
            print(f"Subject: {sub}; Session: {ses}")
            ses_path = os.path.join(sub_path, ses)
            # print(ses_path)
            file_path_list = sorted(glob.glob(os.path.join(ses_path, "*.nii.gz")))
            file_list = sorted(
                [os.path.basename(file_path) for file_path in file_path_list]
            )
            print(file_list)
            # Test if Files of Interest Available
            if len(file_path_list) == 1 and os.path.isfile(file_path_list[0]):
                tissue_class_path = file_path_list[0]
                print("YES")
                try:
                    # Create Mesh for Reference and Processing
                    iso, iso_d = seg_to_mesh(
                        tissue_class_path
                    )  # isosurface & isosurface discrete
                    # Reference Mesh
                    ref_mesh = vedo_to_pv_mesh(iso)  # reference mesh with no processing

                    # Mesh Correction
                    pv_mesh = mesh_fix(iso_d)
                    # # Mesh Processing
                    iso_d = process_mesh(pv_mesh)
                    # Basic Mesh
                    basic_mesh = vedo_to_pv_mesh(iso_d)
                    # basic_mesh.save('/Users/apenkaus/Downloads/0001000.stl')

                except:
                    print(f"{sub} {ses} an exception occurred in mesh generation")
                    temp_list = [ses, "Mesh Generation Error"]
                    error_dict[sub] = temp_list

                # Save Meshes
                # Destination Path
                destination_path = os.path.join(dst_dir_path, sub, ses, "anat")
                # Create Directories
                os.makedirs(destination_path, exist_ok=True)
                mesh_files = [
                    ref_mesh,
                    basic_mesh,
                ]

                # Destination Tags
                dst_tags = [
                    "ref",
                    "bsc",
                ]  # reference, basic processing, 
                
                # Iterate and Save All Meshes
                for mesh_idx, mesh in enumerate(mesh_files):
                    # File Destination Path
                    file_dst_path = (
                        f"{destination_path}/{sub}_lat_vent_{dst_tags[mesh_idx]}.stl"
                    )
                    # Save File
                    mesh.save(file_dst_path)

                # Iterate and Save All Pyvista Meshes
                for mesh_idx, mesh in enumerate(mesh_files):

                    # List of Meshes to Quantify
                    mesh_to_quantify = [
                        "bsc",
                    ]

                    if (
                        dst_tags[mesh_idx] in mesh_to_quantify
                    ):  # select only smoothed and subdivided & decimated meshes to process

                        ########################
                        #### Perturbation ######
                        ########################
                        # Perturbation Parameters
                        mean = 0

                        # Set Perturbation Strenght
                        sigmas = np.array(
                            [
                                0.0,
                                0.00001,
                                0.00005,
                                0.0001,
                                0.0005,
                                0.001,
                                0.005,
                                0.01,
                                0.05,
                                0.1,
                                0.5,
                            ]
                        )
                        # Setup Generator with Fixed Seed
                        rng = np.random.default_rng(
                            12345
                        )  # random number generator with specified seed

                        # Iterate Over Different Strengts of Perturbation
                        for sigma in sigmas:
                            # Perturbation ID
                            ptb = f"ptb={'{:.1e}'.format(sigma)}"
                            print(ptb)

                            # Dublicate Mesh
                            ptb_mesh = mesh.copy()

                            # Original Signal
                            original_signal = ptb_mesh.points

                            # Normalize the signal to [0, 1] range
                            # This ensures a noise SD of 0.5 is meaningful relative to the data
                            data_min, data_max = (
                                original_signal.min(),
                                original_signal.max(),
                            )
                            normalized_signal = (original_signal - data_min) / (
                                data_max - data_min
                            )

                            # Generate Gaussian white noise (Mean=0, SD=sigma)
                            ptb_noise = rng.normal(
                                loc=mean, scale=sigma, size=normalized_signal.shape
                            )

                            # Add noise to the normalized signal
                            perturbed_signal_norm = normalized_signal + ptb_noise

                            # De-normalize to return to original scale
                            perturbed_signal = (
                                perturbed_signal_norm * (data_max - data_min) + data_min
                            )

                            # Introduce noise to mesh, perturbate Mesh
                            ptb_mesh.points = perturbed_signal

                            # # Generate Noise
                            # ptb_noise = rng.normal(loc=mean, scale=sigma, size=ptb_mesh.points.shape)
                            # # Perturbate Mesh
                            # ptb_mesh.points = ptb_mesh.points + ptb_noise

                            # Perturbation Destination Path
                            perturbation_destination_path = os.path.join(
                                dst_dir_path, sub, ses, "anat", "perturbated"
                            )
                            # Create Directories
                            os.makedirs(perturbation_destination_path, exist_ok=True)
                            # Perturbated File Path
                            prtrb_file_dst_path = f"{perturbation_destination_path}/{sub}_lat_vent_{str(ptb).replace('.','')}.stl"
                            # Save Mesh
                            ptb_mesh.save(prtrb_file_dst_path)
                            # Convert PyVista to TriMesh
                            tri_mesh = pv_to_tri(ptb_mesh)

                            # Extract Features ###

                            #  Global Features
                            SA = tri_mesh.area  # surface area
                            V = tri_mesh.volume  # volume
                            SAVR = SA / V  # surface area to volume ratio
                            IMC = (
                                tri_mesh.integral_mean_curvature
                            )  # Integral Mean Curvature

                            # Capture Asphericity
                            # Generate points
                            pts = ptb_mesh.points
                            # Find the best fitting ellipsoid to the points
                            elli = pca_ellipsoid(
                                pts, pvalue=0.95
                            )  #  https://vedo.embl.es/docs/vedo/pointcloud.html#pca_ellipsoid
                            AS = elli.asphericity()  # asphericity
                            ASE = elli.asphericity_error()  # error on asphericity

                            # Capture Convexity and Geometry Features

                            # Compute Shape volume to Convex Hull volume ratio
                            CxVR = (
                                V / tri_mesh.convex_hull.volume
                            )  # Volume devided by convex hull volume
                            # Compute Shape area to Convex Hull area ratio
                            CxSAR = SA / tri_mesh.convex_hull.area
                            # Compute Convex Hull volume
                            CxV = tri_mesh.convex_hull.volume
                            # Compute Convex Hull area
                            CxSA = tri_mesh.convex_hull.area
                            # Compute Convex Hull area to volume ratio
                            CxSAVR = CxSA / CxV

                            # Combine All GLobal and Local Measurments in One List
                            ft_list = [
                                SA,
                                V,
                                SAVR,
                                IMC,
                                AS,
                                ASE,
                                CxV,
                                CxSA,
                                CxSAVR,
                                CxSAR,
                                CxVR,
                            ]

                            # Gaussian Curvatuire at Different Radii
                            gauss_curvature = gauss_curv(tri_mesh, radii_range)

                            # Add Data to DataFrame

                            # Subject Data
                            data = [sub, ses, ptb]  # subject & session & mesh tag
                            data.extend(gauss_curvature)  # extend with radii measurment
                            data.extend(
                                ft_list
                            )  # extend with global and local measurment

                            # Temporary Dictionary
                            temp_dict = dict(zip(cols, data))
                            # Temporary DataFrame
                            temp_df = pd.DataFrame.from_dict(
                                temp_dict, orient="index"
                            ).T
                            # Concatinate Data with Main DataFrame
                            data_frame = pd.concat([data_frame, temp_df])
                            # Save Feature Data
                            data_frame.to_csv(
                                feature_df_path
                            )

