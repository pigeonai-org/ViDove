
#  the entry of the evaluation module
from gemini_batch_processing_task import GeminiEvaluator
from batch_processing_translation_task import batch_process_videos,setup_logging,load_config
from utils.identify_dirty_data import DirtyDataProcessor, dirty_data_list
from generate_eval_result import generate_eval_result
from scores.multi_scores import cal_all_scores
from evaluation.evaluate import load_data
from evaluation.utils.cal_avg_scores_from_results import CalAvgScoresInCsv

def main():
    """
    首先，你要在evaluation/test_data下，有text_data_test.en, text_data_test.zh, text_data_test.id这三个文件
    以及在evaluation/test_data/videos下，有视频文件(数据集)
    """
    
    input_dir = "./evaluation/test_data/videos/"
    output_dir = "./evaluation/test_data/gemini_results"
    

    # 如果你需要测gemini
    translator = GeminiEvaluator(input_dir=input_dir,output_dir=output_dir)
    translator.batch_process_videos()
    
    
    # 如果你要测ViDove
    logger = setup_logging()
    task_cfg = load_config()
    
    task_cfg["output_type"]["video"] = False
    task_cfg["output_type"]["subtitle"] = "srt"
    batch_process_videos(input_dir, output_dir, task_cfg, logger)
    
    # 测完了之后，你应该会在你的output_dir（就是"./evaluation/test_data/gemini_results"）看到很多srt文件
    
    # 如果是测BigVideo数据集的话，你可以运行identify_dirty_data.py，来识别脏数据，并清理脏数据（将脏数据设置为"。"，因为其他模块会对"。"进行判定处理）
    dirty_data_processor = DirtyDataProcessor("./evaluation/test_data/srt_output")
    # dirty_data_list是import进来的，是脏数据列表
    dirty_data_processor.clean_dirty_data(dirty_data_list)
    
    
    # （如果你要测BigVideo数据集的话）然后就可以通过你刚才的得到的大量srt文件，来生成一个eval_result.zh文件了
    # 这个eval_result.zh文件会像是这样，每行对应数据集里一个视频的全部字幕（脏数据会被标记为。）
    """
    他们确实有一个叫做倒车器的部件，所以他们可以从烧胎中倒车。
    。
    但是，这里还有另一个与这个低压场的相互作用，可能比你预期的要更远。
    ...
    """
    generate_eval_result(id_file="./evaluation/test_data/text_data_test.id",
                output="./evaluation/test_data/eval_result.zh",
                srt_dir="./evaluation/test_data/srt_output") 


    # 在你有数据集结果以及数据集答案后，去评测分数
    src_list, mt_list, ref_list = load_data("./evaluation/test_data/text_data_test.en", "./evaluation/test_data/eval_result.zh", "./evaluation/test_data/text_data_test.zh")
    cal_all_scores(src_list, mt_list, ref_list)
    
    # 测出的分数应该会在test_data目录下生成一个result.csv文件，测一下其中的平均值
    cal_avg_scores_in_csv = CalAvgScoresInCsv('./evaluation/test_data/result.csv')
    cal_avg_scores_in_csv.cal_avg_scores()
    
    
  
    

if __name__ == "__main__":
    main()
