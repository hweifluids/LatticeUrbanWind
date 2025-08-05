
import vtk
import os,sys
from pathlib import Path
import xarray as xr 
import numpy as np
from vtk.util import numpy_support
import matplotlib.pyplot as plt

# -------- 1. read data --------
conf_file = Path(__file__).parent.parent / "conf.txt"
if conf_file.exists():
    import re
    txt = conf_file.read_text()
    casename = re.search(r"casename\s*=\s*([^\s]+)", txt).group(1)
    print(f"Loaded conf.txt, case={casename}")
else:
    print("No conf.txt found: use provided nc_path and no cropping")

# --- if conf.txt gives cut_lon / cut_lat, then perform cut ------------------
m_lon = re.search(r"cut_lon_manual\s*=\s*\[([^\]]+)\]", txt)
m_lat = re.search(r"cut_lat_manual\s*=\s*\[([^\]]+)\]", txt)
if m_lon and m_lat and m_lon.group(1).strip() and m_lat.group(1).strip():
    min_lon, max_lon = [float(v) for v in m_lon.group(1).split(",")]
    min_lat, max_lat = [float(v) for v in m_lat.group(1).split(",")]
    print(f"Crop to conf range: lon {min_lon}-{max_lon}, "
          f"lat {min_lat}-{max_lat}")

path = "/gpu/hanshuai/fluent3D/LBMcpWRF/WRFcpLBM_v2/caseData"
# 文件路径
wind_file = os.path.join(path, casename, f"uvw-{casename}.vtk")
output_nc = os.path.join(path, casename, f"uvw-{casename}.nc")

# ==========================
# 创建读取器
reader = vtk.vtkDataSetReader()
reader.SetFileName(wind_file)
reader.Update()  # 执行读取操作

# 获取数据
data = reader.GetOutput()

# 获取原始数据的维度、原点和间距
dims = data.GetDimensions()
origin = data.GetOrigin()
spacing = data.GetSpacing()
print(f"data dims: {dims}")
print(f"data origin: {origin}")
print(f"data spacing: {spacing}")
# vtk_to_numpy 进行数据转换
pointdata = data.GetPointData()
print("Number Of ArrayS:", pointdata.GetNumberOfArrays())
dataArray = numpy_support.vtk_to_numpy(pointdata.GetScalars())

# 重塑数组以匹配 VTK 数据的维度
# VTK 使用 (X,Y,Z) 顺序，而 NumPy 使用 (Z,Y,X) 顺序
u = dataArray[:,0].reshape(dims[2], dims[1], dims[0])
v = dataArray[:,1].reshape(dims[2], dims[1], dims[0])
w = dataArray[:,2].reshape(dims[2], dims[1], dims[0])

# 如果需要 C 风格的索引顺序 (X,Y,Z)，可以转置
# u = u.transpose(2, 1, 0)

# 根据 origin, spacing, dims 信息生成 x,y,z 坐标信息
origin_tran = (0, 0, 0)
max_x = origin_tran[0] + dims[0]*spacing[0]
x = np.linspace(origin_tran[0], max_x, dims[0])

max_y = origin_tran[1] + dims[1]*spacing[0]
y = np.linspace(origin_tran[1], max_y, dims[1])

max_z = origin_tran[2] + dims[2]*spacing[2]
z = np.linspace(origin_tran[2], max_z, dims[2])

z = z-50
idx = np.where((z>=0)& (z<=300))
start = idx[0].min()
end = idx[0].max()
# 下面有一个地基高度z，需要减去50
u = u[start:end]
v = v[start:end]
w = w[start:end]
h = z[start:end]
print(f"h: {h}")

ny = len(y)
nx = len(x)
lon_s = np.linspace(min_lon, max_lon, nx)
lat_s = np.linspace(min_lat, max_lat, ny)

# 按照经纬度范围进行数据裁切
# idx_lon = np.where((lon_s>=113.6432)&(lon_s<=113.6812))
# idx_lat = np.where((lat_s>=34.6905)&(lat_s<=34.7227))
# c_lon_s = idx_lon[0].min()
# c_lon_e = idx_lon[0].max()
# c_lat_s = idx_lat[0].min()
# c_lat_e = idx_lat[0].max()

# lon_s = lon_s[c_lon_s:c_lon_e]
# lat_s = lat_s[c_lat_s:c_lat_e]
# u = u[:, c_lat_s:c_lat_e, c_lon_s:c_lon_e]
# v = v[:, c_lat_s:c_lat_e, c_lon_s:c_lon_e]
# w = w[:, c_lat_s:c_lat_e, c_lon_s:c_lon_e]

# 数据存储为nc文件
vars_coords = {
    # "x": (("x",), x, {"long_name": 'x', "units": 'meters'}),
    # "y": (("y",), y, {"long_name": 'y', "units": 'meters'}),
    "h": (("h",), h, {"long_name": 'h', "units": 'meters'}),
    "lat": (("lat"), lat_s, {"long_name": 'latitude', "units": 'degree_north'}),
    "lon": (("lon"), lon_s, {"long_name": 'longitude', "units": 'degree_east'}),
    "u":(("h","lat","lon"), u, {"long_name": 'uWind', "units": 'degree_east'}),
    "v":(("h","lat","lon"), v, {"long_name": 'vWind', "units": 'degree_east'}),
    "w":(("h","lat","lon"), w, {"long_name": 'wWind', "units": 'degree_east'}),
    # "topo":(("h","lat","lon"), topo, {"long_name": 'elevation', "units": 'degree_east'}),
}
encoding = {
    # "x": {'dtype': 'float32', 'zlib': True, 'complevel': 3, '_FillValue': np.nan, },
    # "y": {'dtype': 'float32', 'zlib': True, 'complevel': 3, '_FillValue': np.nan, },
    "lat": {'dtype': 'float32', 'zlib': True, 'complevel': 3, '_FillValue': np.nan, },
    "lon": {'dtype': 'float32', 'zlib': True, 'complevel': 3, '_FillValue': np.nan, },
    "u": {'dtype': 'float32', 'zlib': True, 'complevel': 3, '_FillValue': np.nan, },
    "v": {'dtype': 'float32', 'zlib': True, 'complevel': 3, '_FillValue': np.nan, },
    "w": {'dtype': 'float32', 'zlib': True, 'complevel': 3, '_FillValue': np.nan, },
    # "topo": {'dtype': 'float32', 'zlib': True, 'complevel': 3, '_FillValue': np.nan, },
}
print(len(h))
print(u.shape)
out_xr = xr.Dataset(
    data_vars=vars_coords,
)
out_xr.attrs['description'] = 'NetCDF created from vkt file (TJMJ)'
out_xr.attrs['projection'] = ""
out_xr.attrs['transfrom'] = ""
out_xr.to_netcdf(
    output_nc,
    mode="w",
    format="NETCDF4",
    encoding=encoding
)