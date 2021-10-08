import torch

from multiplexer.structures.word_result import WordResult


def run_model(model, images, cfg):
    prediction_dict = {
        "global_prediction": None,
    }

    cpu_device = torch.device("cpu")

    with torch.no_grad():
        predictions, proposals, seg_results_dict = model(images)
        if cfg.MODEL.TRAIN_DETECTION_ONLY:
            prediction_dict["global_prediction"] = [o.to(cpu_device) for o in predictions]
            assert len(seg_results_dict["scores"]) == 1
            prediction_dict["scores"] = seg_results_dict["scores"][0].to(cpu_device).tolist()
            # Add dummy word result list
            word_result_list = []
            for _ in range(len(prediction_dict["scores"])):
                word_result = WordResult()
                word_result.seq_word = ""
                word_result_list.append(word_result)
            prediction_dict["word_result_list"] = word_result_list
        else:
            prediction_dict["global_prediction"] = [o.to(cpu_device) for o in predictions[0]]

            prediction_dict["word_result_list"] = predictions[1]["word_result_list"]

    return prediction_dict
