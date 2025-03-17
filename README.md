# TransformerUpscaler
### ML@SJSU

## Transformer Model Architecture
![Transformer Model Architecture](resources/architecture.png)

## Train
`python train.py --data_dir images/training_set`

## Inference (single image)
`python inference.py --image_path images/training_set/image_0.jpg --out_res 1080`

## Results Comparison

<table>
  <tr>
    <th style="text-align:center;">Low Resolution Input (1280×720)</th>
    <th style="text-align:center;">Upscaled Output (1920×1080)</th>
  </tr>
  <tr>
    <td style="text-align:center;">
      <div style="text-align:center;">
        <img src="resources/demo/input_1.png" alt="Low Resolution Image" style="width:400px;">
        <p>Original</p>
      </div>
    </td>
    <td style="text-align:center;">
      <div style="text-align:center;">
        <img src="resources/demo/output_1.png" alt="Upscaled Image" style="width:400px;">
        <p>Upscaled image</p>
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      <div style="text-align:center;">
        <img src="resources/demo/input_2.png" alt="Low Resolution Image" style="width:400px;">
        <p>Original</p>
      </div>
    </td>
    <td style="text-align:center;">
      <div style="text-align:center;">
        <img src="resources/demo/output_2.png" alt="Upscaled Image" style="width:400px;">
        <p>Upscaled image</p>
      </div>
    </td>
  </tr>
</table>