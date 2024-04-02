def lossPlayerFunction(
        predictions, 
        target, 
        W=16,               # W, H: Width and height of the grid.
        H=8, 
        B=3,                # B, C: Number of bounding boxes per grid cell, and number of classes.
        C=3, 
        lambda_coord=5,     # lambda_coord: Weight for the coordinate loss component.
        lambda_noobj=0.5,
        lambda_orientation = 1   # Orientation hyper parametr.
    ):


    batch_size = predictions.shape[0]
    
    # batch structure [100, 8, 16, 3, 8] -> [Batch_size, grid_H, grid_W, B, Box]
    # Box structure [Confident, x, y, radius, orientation, [one, hot, class]]
    
    predictions = resortPredictionByTargetAreaRatio(target, predictions)

    # Extract components from predictions and target
    pred_boxes = predictions[..., :5]
    pred_classes = predictions[..., 5:]
    
    target_boxes = target[..., :5]
    target_classes = target[..., 5:]
    
    # Object mask where target confidence is 1 (object present)
    # No-object mask where target confidence is 0 (no object present)
    obj_mask = target_boxes[..., 0] == 1
    no_obj_mask = target_boxes[..., 0] == 0
  
    # X, Y loss for the Coordinates of the center of boxes
    coord_loss = lambda_coord * torch.sum(
        (pred_boxes[obj_mask][..., 1:3] - target_boxes[obj_mask][..., 1:3])**2
    )

    # RADIUS for the dimensions of boxes (sqrt is applied to emphasize smaller boxes)
    pred_signs = torch.sign(pred_boxes[..., 3])
    radius_loss = lambda_coord * torch.sum(
        (target_boxes[obj_mask][..., 3].sqrt() - pred_signs * pred_boxes[obj_mask][..., 3].abs().sqrt())**2
    )

    # ORIENTATION loss for the Coordinates of the center of boxes
    orientation_loss = lambda_orientation * torch.sum(
        (pred_boxes[obj_mask][..., 4] - target_boxes[obj_mask][..., 4])**2
    )

    # OBJECT HERE loss - only for boxes that are responsible for detecting objects
    object_exist_loss = torch.sum(
        (pred_boxes[obj_mask][..., 0] - target_boxes[obj_mask][..., 0])**2
    )

    # NO OBJECT loss - for boxes that are not responsible for detecting objects
    # Typically, this has a lower weight (lambda_noobj) because there are many more negative (no-object) examples
    object_not_exist_loss = lambda_noobj * torch.sum(
        (pred_boxes[no_obj_mask][..., 0] - target_boxes[no_obj_mask][..., 0])**2
    )
    
    # CLASS loss for the class probabilities of each grid cell
    class_loss = torch.sum(
        (pred_classes - target_classes)**2
    )
    
    # Total loss
    total_loss = coord_loss 
    # + radius_loss 
    # + orientation_loss 
    + object_exist_loss 
    + object_not_exist_loss 
    + class_loss
    return total_loss / batch_size  # Normalize by batch size for consistency

# loss = lossPlayerFunction(outputs, batch_annotations, device)
# print(loss.item())