
import vtk
import os,sys
from pathlib import Path
import xarray as xr 
import numpy as np
from vtk.util import numpy_support
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer

# -------- 1. read data --------
conf_file = Path(__file__).parent.parent / "conf.txt"
if conf_file.exists():
    import re
    txt = conf_file.read_text()
    casename = re.search(r"casename\s*=\s*([^\s]+)", txt).group(1)
    datetime = re.search(r"datetime\s*=\s*([^\s]+)", txt).group(1)   # lwg: datetime
    print(f"Loaded conf.txt, case={casename}, datetime={datetime}, ")
else:
    print("No conf.txt found: use provided nc_path and no cropping")

# --- if conf.txt gives cut_lon / cut_lat, then perform cut ------------------
m_x = re.search(r"si_x\s*=\s*\[([^\]]+)\]", txt)
m_y = re.search(r"si_y\s*=\s*\[([^\]]+)\]", txt)
if m_x and m_y and m_x.group(1).strip() and m_y.group(1).strip():
    min_x, max_x = [float(v) for v in m_x.group(1).split(",")]
    min_y, max_y = [float(v) for v in m_y.group(1).split(",")]
    print(f"Crop to conf range: x= {min_x}-{max_x}, y= {min_y}-{max_y}")

# 文件路径
wind_file = Path("caseData")/casename/f"uvw-{casename}_{datetime}.vtk"
output_nc = Path("caseData")/casename/f"uvw-{casename}_{datetime}.nc"

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
# origin_tran = (0, 0, 0)
# max_x = origin_tran[0] + dims[0]*spacing[0]
# x = np.linspace(origin_tran[0], max_x, dims[0])
# max_y = origin_tran[1] + dims[1]*spacing[0]
# y = np.linspace(origin_tran[1], max_y, dims[1])

max_z = 0 + (dims[2]-1)*spacing[2]
z = np.linspace(0, max_z, dims[2])
print(f"z: {z}")

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

# ——— project from UTM Zone 50N and obtain range ———
xs = np.linspace(min_x, max_x, dims[0])
ys = np.linspace(min_y, max_y, dims[1])

xx, yy = np.meshgrid(xs, ys)
# points = [Point(x, y) for x, y in zip(xx, yy)]
# gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:32650").to_crs(epsg=4326)
# 创建转换器（UTM -> WGS84）
transformer = Transformer.from_crs(
    crs_from="EPSG:32650",
    crs_to="EPSG:4326",  # WGS84经纬度
    always_xy=True  # x为经度，y为纬度
)
# 批量转换坐标
lons, lats = transformer.transform(xx.flatten(), yy.flatten())
lons  = lons.reshape((dims[1], dims[0]))
lats  = lats.reshape((dims[1], dims[0]))

# 数据存储为nc文件
vars_coords = {
    "x": (("x",), xs, {"long_name": 'x', "units": 'meters'}),
    "y": (("y",), ys, {"long_name": 'y', "units": 'meters'}),
    "h": (("h",), h, {"long_name": 'h', "units": 'meters'}),
    "lat": (("y","x"), lats, {"long_name": 'latitude', "units": 'degree_north'}),
    "lon": (("y","x"), lons, {"long_name": 'longitude', "units": 'degree_east'}),
    "u":(("h","y","x"), u, {"long_name": 'uWind', "units": 'degree_east'}),
    "v":(("h","y","x"), v, {"long_name": 'vWind', "units": 'degree_east'}),
    "w":(("h","y","x"), w, {"long_name": 'wWind', "units": 'degree_east'}),
    # "topo":(("h","y","x"), topo, {"long_name": 'elevation', "units": 'degree_east'}),
}
encoding = {
    "x": {'dtype': 'float32', 'zlib': True, 'complevel': 3, '_FillValue': np.nan, },
    "y": {'dtype': 'float32', 'zlib': True, 'complevel': 3, '_FillValue': np.nan, },
    "lat": {'dtype': 'float32', 'zlib': True, 'complevel': 3, '_FillValue': np.nan, },
    "lon": {'dtype': 'float32', 'zlib': True, 'complevel': 3, '_FillValue': np.nan, },
    "u": {'dtype': 'float32', 'zlib': True, 'complevel': 3, '_FillValue': np.nan, },
    "v": {'dtype': 'float32', 'zlib': True, 'complevel': 3, '_FillValue': np.nan, },
    "w": {'dtype': 'float32', 'zlib': True, 'complevel': 3, '_FillValue': np.nan, },
    # "topo": {'dtype': 'float32', 'zlib': True, 'complevel': 3, '_FillValue': np.nan, },
}
print(u.shape)
out_xr = xr.Dataset(
    data_vars=vars_coords,
)
out_xr.attrs['description'] = 'NetCDF created from vkt file (TJMJ)'
out_xr.attrs['Projection'] = "utm50n"
out_xr.attrs['transfrom'] = ""
out_xr.to_netcdf(
    output_nc,
    mode="w",
    format="NETCDF4",
    encoding=encoding
)
print("Saved translated file: ", output_nc)