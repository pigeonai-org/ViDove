from remove_timestamp import RemoveTimestampConverter

class ViDoveDataSetAdapter():
    def __init__(self, input_dir=None, output_dir=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
    def batch_processing_with_vidove(self):
        pass
    
    def batch_processing_with_gemini(self):
        pass
    
    def batch_formatting(self):
        
        # 单个文件处理
        converter = RemoveTimestampConverter(
            single_file=r"evaluation\test_data\28e6c89c-2a04-45d9-8fdb-6b4ed23f6087_ZH.srt", # for single file processing
            output_file=r"evaluation\test_data\remove_timestamp_result.txt",
        )
        # converter.process_single_file()
        
        # 批量处理
        converter = RemoveTimestampConverter(
            id_list=r"evaluation\test_data\text_data_test.id",
            target_srt_dir=r"evaluation\test_data\test\srt_output",
            output_dir=r"evaluation\test_data\vidove_result",
        )
        converter.process_id_list_to_separate_files()

if __name__ == "__main__":
    adapter = ViDoveDataSetAdapter(
        input_dir=r"evaluation\test_data\test\srt_output",
        output_dir=r"evaluation\test_data\test\txt_output",
    )
    # 先批量把视频都翻译了
    adapter.batch_processing_with_vidove()
    # 再批量把翻译的srt文件对齐format
    adapter.batch_formatting()
    # 
