from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def print_results(coco_eval: COCOeval) -> None:
    """Evaluates, accumulates, and summarizes the COCO evaluation results.

    Args:
        coco_eval (COCOeval): The COCO evaluation object.
    """
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def evaluate_coco_results(gt_path: str, pred_path: str) -> None:
    """Evaluates COCO results for all categories and per category.

    Args:
        gt_path (str): Path to the ground truth JSON file.
        pred_path (str): Path to the prediction JSON file.
    """
    coco = COCO(gt_path)
    pred_coco = coco.loadRes(pred_path)
    categories = coco.cats

    print("-------------------------------------------------------------------------------")
    print("CATEGORIES:")
    print(categories)
    print("-------------------------------------------------------------------------------")

    coco_eval = COCOeval(cocoGt=coco, cocoDt=pred_coco, iouType="bbox")

    # Evaluate for all classes
    print("ALL CLASSES:")
    print_results(coco_eval)

    # Evaluate per class
    for value in categories.values():
        category_id = value["id"]
        class_name = value["name"]
        print("-------------------------------------------------------------------------------")
        print("CLASS_NAME = ", class_name)

        coco_eval.params.catIds = [category_id]
        print_results(coco_eval)
