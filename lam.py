from torch import tensor
from lam.utils import *
from lam.backprop import *
from lam.attributes import *
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import importlib
import argparse
from tools.utils import get_latest_checkpoint

def pick_coordinates(image, num_points=1):
    """
    Displays the image and allows the user to click on it to select coordinates.

    Parameters:
      image: A 2D (grayscale) or 3D (color) numpy array representing the image.
      num_points: The number of points to select.

    Returns:
      points: A list of (x, y) coordinates where the user clicked.
    """
    plt.imshow(image, cmap='gray')
    plt.title("Click on the image to select coordinates")
    # ginput returns a list of (x, y) tuples
    points = plt.ginput(num_points, timeout=0)
    plt.close()
    return points


def main():
    parser = argparse.ArgumentParser(description="LAM: Local Attention Map")
    parser.add_argument("--model", type=str, default="Fastv2",
                        help="Model name")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--out", type=str, default="lam.jpg",
                        help="Output image path")
    parser.add_argument("--window_size", type=int, default=50,
                        help="Window size for LAM")
    parser.add_argument("--scale", type=int, required=True,
                        help='Scale factor for model')
    parser.add_argument("--w", type=int, default=200,
                        help="X coordinate for LAM")
    parser.add_argument("--h", type=int, default=200, 
                        help="Y coordinate for LAM")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing model checkpoints (default: models/{model}/checkpoints/)")
    
    args = parser.parse_args()
    
    img_hr = Image.open(args.image_path).convert("RGB")
    img_lr = img_hr.resize((img_hr.size[0] // args.scale, img_hr.size[1] // args.scale), Image.BICUBIC)
    tensor_lr = PIL2Tensor(img_lr)[:3] ; tensor_hr = PIL2Tensor(img_hr)[:3]
    cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 2) ; cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 2)
    
    coords = pick_coordinates(cv2_hr, num_points=1)
    print(coords) 
    w = int(coords[0][0])
    h = int(coords[0][1]) 
    print(f"Selected coordinates: ({w}, {h})")
    
    draw_img = pil_to_cv2(img_hr)
    cv2.rectangle(draw_img, (w, h), (w + args.window_size, h + args.window_size), (0, 0, 255), 2)
    position_pil = cv2_to_pil(draw_img)
         
    model_module = importlib.import_module(f"models.{args.model}.model")
    TransformerModel = model_module.TransformerModel

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    model = TransformerModel().to(device=device)
    
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"models/{args.model}/checkpoints"
    checkpoint_path, _ = get_latest_checkpoint(args.checkpoint_dir)
    print(f'Loading checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()

    sigma = 1.2
    fold = 50
    l = 9
    alpha = 0.5
    attr_objective = attribution_objective(attr_grad, h, w, window=args.window_size)
    gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lr.numpy(), model, attr_objective,
                                                                              gaus_blur_path_func, upscale_factor=args.scale)
    grad_numpy, result = saliency_map_PG(interpolated_grad_numpy, result_numpy)
    abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
    saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=args.scale)
    saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)
    blend_abs_and_input = cv2_to_pil(pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    blend_kde_and_input = cv2_to_pil(pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    pil = make_pil_grid(
        [position_pil,
        saliency_image_abs,
        blend_abs_and_input,
        blend_kde_and_input,
        Tensor2PIL(torch.clamp(torch.from_numpy(result), min=0., max=1.))]
    )
    
    pil.save(args.out)
    pil.show()
    
    
if __name__ == '__main__':
    main()