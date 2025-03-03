import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

default_read_legth = 2 ** 32

def showImage(color:torch.tensor):
    """_summary_ (3, x, y) tensor -> image

    Args:
        color (torch.tensor): _description_

    Returns:
        _type_: _description_
    """
    color = color.permute(1, 2, 0).to('cpu')
    print("image size:", color.shape[0], "X", color.shape[1], "channels:", color.shape[2])
    color_image = Image.fromarray((color.numpy() * 255).astype('uint8'))
    return color_image

def showDepth(color:torch.tensor):
    """_summary_ (x, y) tensor -> image

    Args:
        color (torch.tensor): _description_

    Returns:
        _type_: _description_
    """
    color = color.to('cpu').numpy()
    color[color==-1.0] = -10.0
    height, width = color.shape
    dpi = 50
    figsize = width / float(dpi), height / float(dpi)
    plt.figure(figsize=figsize)

    depth_map = plt.imshow(color, cmap='viridis', vmin=color.min(), vmax=color.max())
    plt.colorbar(depth_map, label='Depth')

    plt.title('Depth Map')
    plt.show()
    
def printInfoArray(array:np.array):
    print("\tshape:", array.shape[0], "\tmin:", array.min(), "\tmax:", array.max(), "\tmean:", array.mean())
    num_nan = int(np.isnan(array).sum())
    num_inf = int(np.isinf(array).sum())
    if num_nan != 0:
        print("-- NAN:", num_nan)
    if num_inf != 0:
        print("-- INF:", num_inf)
    
def loadInt(filename, size=default_read_legth):
    data = np.fromfile(filename, dtype=np.int32, count=size)
    print(filename, end='\t')
    printInfoArray(data)
    return data

def loadInt8(filename, size=default_read_legth):
    data = np.fromfile(filename, dtype=np.int8, count=size)
    print(filename, end='\t')
    printInfoArray(data)
    return data

def loadFloat(filename, size=default_read_legth):
    data = np.fromfile(filename, dtype=np.float32, count=size)
    print(filename, end='\t')
    printInfoArray(data)
    return data

def loadBool(filename, size=default_read_legth):
    data = np.fromfile(filename, dtype=np.bool, count=size)
    print(filename, end='\t')
    printInfoArray(data)
    return data

def printInfoTensor(array: torch.tensor):
    num_nan = array.isnan().sum()
    num_inf = array.isinf().sum()
    print(
        "shape:", array.shape, 
        "\tmin:", float(array.min()), 
        "\tmax:", float(array.max()), 
        "\tmean:", float(array.mean()), 
    )
    if num_nan != 0:
        print("-- NAN:", float(num_nan))
    if num_inf != 0:
        print("-- INF:", float(num_inf))
    print()
    
def loadTensorC(filename):
    """_summary_ read `.pt` which stored by C++

    Args:
        filename (_type_): _description_

    Returns:
        _type_: _description_  torch.tensor
    """
    print("---", filename.split('/')[-1])
    a = torch.jit.load(filename)
    for name, param in a.named_parameters():
        res = param
    printInfoTensor(res)
    return res

def load_args(arg_names, base_path):
    args = {}
    for arg_name in arg_names:
        args[arg_name] = torch.load(base_path.format(arg_name), weights_only=True)
        try:
            num_nan = args[arg_name].isnan().sum()
            num_inf = args[arg_name].isinf().sum()
            num_nan = int(num_nan)
            num_inf = int(num_inf)
            if num_nan != 0:
                print(arg_name, "num nan:", num_nan)
            if num_inf != 0:
                print(arg_name, "num inf:", num_inf)
        except:
            None
    print(args.keys())
    return args

def loadForward(base_path="test_data/forward_tensors"):
    arg_names_forward = [
            'sigma', 
            'weight_background', 
            'anchors3D',
            'anchor_scales',
            'bg', 
            'means3D', 
            'colors_precomp', 
            'opacities', 
            'opacity_weight', 
            'scales', 
            'rotations', 
            'scale_modifier', 
            'cov3Ds_precomp', 
            'viewmatrix', 
            'projmatrix', 
            'tanfovx', 
            'tanfovy', 
            'image_height', 
            'image_width', 
            'sh', 
            'sh_degree', 
            'campos', 
            'prefiltered', 
            'debug'
        ]
    return load_args(arg_names_forward, base_path + "/{}.pt")

def loadBackward():
    arg_names_backward = [
        "sigma",
        "error_map",
        "render_image",
        "opacity_weight",
        "opacities",
        "bg",
        "means3D",
        "radii",
        "colors_precomp",
        "scales",
        "rotations",
        "scene_radii",
        "scale_modifier",
        "cov3Ds_precomp",
        "viewmatrix",
        "projmatrix",
        "tanfovx",
        "tanfovy",
        "grad_out_color",
        "sh",
        "sh_degree",
        "campos",
        "geomBuffer",
        "num_rendered",
        "binningBuffer",
        "imgBuffer",
        "debug"
    ]
    return load_args(arg_names_backward, "test_data/backward_tensors/{}.pt")

def savePly(array, filename):
    from plyfile import PlyData, PlyElement
    points = np.asarray(array)
    vertex_list = np.array([(x, y, z) for x, y, z in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    # Create a PlyElement for the vertices
    vertex_element = PlyElement.describe(vertex_list, 'vertex')
    # Create a PlyData object and write it to a file
    ply_data = PlyData([vertex_element], text=True)
    ply_data.write(filename)