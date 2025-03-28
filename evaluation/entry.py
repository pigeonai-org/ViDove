
#  the entry of the evaluation module
from gemini_batch_processing_task import GeminiEvaluator
from batch_processing_translation_task import batch_process_videos,setup_logging,load_config
from utils.identify_dirty_data import DirtyDataProcessor
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
    dirty_data_list = ['00MkZbEwoZE_14.srt', '00MkZbEwoZE_20.srt', '0Tkvi_YDlK4_00.srt', '0Tkvi_YDlK4_15.srt', '0UakYLTiplc_00.srt', '0UakYLTiplc_19.srt', '0UakYLTiplc_21.srt', '0UakYLTiplc_24.srt', '3t0IlJtGO88_27.srt', '6a1MWu1MdvE_11.srt', '7f1s1ubpHF4_06.srt', '7f1s1ubpHF4_12.srt', '7f1s1ubpHF4_13.srt', '7f1s1ubpHF4_14.srt', '7taB6hQ7430_38.srt', '8xF_Dfe4rn8_00.srt', '8xF_Dfe4rn8_107.srt', '8xF_Dfe4rn8_137.srt', '8xF_Dfe4rn8_138.srt', '8xF_Dfe4rn8_38.srt', '8xF_Dfe4rn8_46.srt', '8xF_Dfe4rn8_72.srt', '8xF_Dfe4rn8_77.srt', '9zwO6I96oJg_01.srt', '9zwO6I96oJg_24.srt', 'aUN5EdBHHtA_24.srt', 'BbkvWitFkns_48.srt', 'ChnMBUVLhgA_05.srt', 'ChnMBUVLhgA_20.srt', 'CsXGMeMc75U_06.srt', 'CXKVmQYGDu4_12.srt', 'CYz8zVAMpfw_04.srt', 'CYz8zVAMpfw_19.srt', 'D4a_i3rpLzg_01.srt', 'Do51eJowSyg_102.srt', 'Do51eJowSyg_105.srt', 'Do51eJowSyg_113.srt', 'Do51eJowSyg_138.srt', 'Do51eJowSyg_141.srt', 'Do51eJowSyg_144.srt', 'Do51eJowSyg_41.srt', 'Do51eJowSyg_45.srt', 'Do51eJowSyg_58.srt', 'Do51eJowSyg_59.srt', 'Do51eJowSyg_68.srt', 'Do51eJowSyg_86.srt', 'Do51eJowSyg_87.srt', 'Do51eJowSyg_95.srt', 'Do51eJowSyg_98.srt', 'f4KOKfhqp7w_06.srt', 'f4KOKfhqp7w_11.srt', 'GHWxQ9Bphm8_00.srt', 'GM9jTRFCtj8_00.srt', 'hWpbco3F4L4_107.srt', 'hWpbco3F4L4_115.srt', 'hWpbco3F4L4_157.srt', 'hWpbco3F4L4_158.srt', 'hWpbco3F4L4_21.srt', 'hWpbco3F4L4_24.srt', 'hWpbco3F4L4_25.srt', 'IT71qzZw6B8_01.srt', 'Jx2kXT2o_Zw_23.srt', 'mZqpsnPCSIw_10.srt', 'mZqpsnPCSIw_28.srt', 'NEFVpQ66KEo_02.srt', 'NEFVpQ66KEo_29.srt', 'NtKyiGbD6_4_03.srt', 'NtKyiGbD6_4_08.srt', 'NtKyiGbD6_4_16.srt', 'NtKyiGbD6_4_27.srt', 'NtKyiGbD6_4_34.srt', 'QA2HuYGCALQ_09.srt', 'qOzpF9LUw64_05.srt', 'SYjRgxNRDOY_02.srt', 'SYjRgxNRDOY_26.srt', 'SYjRgxNRDOY_27.srt', 'SYjRgxNRDOY_35.srt', 'SYjRgxNRDOY_42.srt', 'SYjRgxNRDOY_45.srt', 'SYjRgxNRDOY_47.srt', 'SYjRgxNRDOY_60.srt', 'SYjRgxNRDOY_73.srt', 'SYjRgxNRDOY_74.srt', 'SYjRgxNRDOY_79.srt', 'UoStHRC43H8_07.srt', 'UoStHRC43H8_16.srt', 'v0294cg10000c6hohc3c77u95on42b7g_171.srt', 'v0294cg10000c6hohc3c77u95on42b7g_376.srt', 'v0294cg10000c6hohc3c77u95on42b7g_49.srt', 'v0294cg10000c6j2qk3c77u8onkj7ihg_161.srt', 'v0294cg10000c6jf36rc77u94potnq90_43.srt', 'v0294cg10000c6jq6ajc77u5ga5iu8c0_121.srt', 'v0294cg10000c6jq6ajc77u5ga5iu8c0_26.srt', 'v0294cg10000c6jq6ajc77u5ga5iu8c0_89.srt', 'v0294cg10000c6k4td3c77u77qrjmbf0_0.srt', 'v0294cg10000c6kt6l3c77ucaaar4kr0_33.srt', 'v0294cg10000c6kt6l3c77ucaaar4kr0_58.srt', 'v0294cg10000c6le7obc77u62sv6k3h0_0.srt', 'v0294cg10000c6le7obc77u62sv6k3h0_39.srt', 'v0294cg10000c6le7obc77u62sv6k3h0_4.srt', 'v0294cg10000c6le7obc77u62sv6k3h0_41.srt', 'v0294cg10000c6le7obc77u62sv6k3h0_8.srt', 'v0294cg10000c6t0j2rc77u6rv3b0p0g_23.srt', 'v0294cg10000c6t0j2rc77u6rv3b0p0g_29.srt', 'v0294cg10000c6t0j2rc77u6rv3b0p0g_30.srt', 'v0294cg10000c74odf3c77uagidi1qf0_2.srt', 'v0294cg10000c7iqslbc77u2p0o1ea80_147.srt', 'v0294cg10000c7rd73rc77u0ehls07kg_39.srt', 'v0294cg10000c7rujq3c77u2vf6ibo60_17.srt', 'v0294cg10000c83si03c77ue1vchkdj0_29.srt', 'v0294cg10000c83si03c77ue1vchkdj0_50.srt', 'v0294cg10000c83si03c77ue1vchkdj0_67.srt', 'v0294cg10000c83si03c77ue1vchkdj0_70.srt', 'v0294cg10000c874u4bc77u5f28dm590_39.srt', 'v0294cg10000c874u4bc77u5f28dm590_96.srt', 'v0294cg10000c87l78jc77u52ggje78g_12.srt', 'v0294cg10000c87l78jc77u52ggje78g_16.srt', 'v0294cg10000c87l78jc77u52ggje78g_21.srt', 'v0294cg10000c87l78jc77u52ggje78g_23.srt', 'v0294cg10000c87l78jc77u52ggje78g_32.srt', 'v0294cg10000c89694jc77u0850rohl0_49.srt', 'v0294cg10000c8rct33c77udjr3hdlug_28.srt', 'v0294cg10000c8rct33c77udjr3hdlug_36.srt', 'v0294cg10000c9cfuprc77ua1qpu2qgg_25.srt', 'v0294cg10000c9igmebc77u5bcl3vtk0_2.srt', 'v0294cg10000c9igmebc77u5bcl3vtk0_29.srt', 'v0294cg10000c9r0sibc77ub3rvsiuj0_3.srt', 'v0294cg10000c9sedlrc77ue02r00pog_38.srt', 'v0294cg10000c9u7kljc77u9hrcmsukg_38.srt', 'v0294cg10000c9u7kljc77u9hrcmsukg_82.srt', 'v0294cg10000cacn6ujc77u3c6j03o70_1.srt', 'v0294cg10000camip7jc77uddhph5qm0_44.srt', 'v0294cg10000cb8j90jc77u338nnc6s0_52.srt', 'v0294cg10000cbb406rc77udk5e968i0_15.srt', 'v0294cg10000cbb406rc77udk5e968i0_31.srt', 'v0394cg10000c40fno3c77u62s93ugrg_1.srt', 'v0394cg10000c40fno3c77u62s93ugrg_2.srt', 'v0394cg10000c40fno3c77u62s93ugrg_23.srt', 'v0394cg10000c40fno3c77u62s93ugrg_3.srt', 'v0394cg10000c40fno3c77u62s93ugrg_32.srt', 'v0394cg10000c6h51erc77u20sog1hgg_25.srt', 'v0394cg10000c6i48ubc77u43t50pgsg_14.srt', 'v0394cg10000c6i48ubc77u43t50pgsg_15.srt', 'v0394cg10000c6i48ubc77u43t50pgsg_22.srt', 'v0394cg10000c6i48ubc77u43t50pgsg_5.srt', 'v0394cg10000c6i48ubc77u43t50pgsg_79.srt', 'v0394cg10000c6iar1rc77ucsqln0brg_146.srt', 'v0394cg10000c6it0crc77u40bd5qfrg_28.srt', 'v0394cg10000c6ji6njc77u8bvf1hb00_1.srt', 'v0394cg10000c6ji6njc77u8bvf1hb00_10.srt', 'v0394cg10000c6ji6njc77u8bvf1hb00_53.srt', 'v0394cg10000c6ji6njc77u8bvf1hb00_6.srt', 'v0394cg10000c6ji6njc77u8bvf1hb00_68.srt', 'v0394cg10000c6ji6njc77u8bvf1hb00_70.srt', 'v0394cg10000c6jjjtrc77u8bkp0uu60_3.srt', 'v0394cg10000c6jjjtrc77u8bkp0uu60_8.srt', 'v0394cg10000c6jjmojc77u03tn4qlpg_140.srt', 'v0394cg10000c6jjmojc77u03tn4qlpg_29.srt', 'v0394cg10000c6jjmojc77u03tn4qlpg_63.srt', 'v0394cg10000c6jlcn3c77u71r64sig0_62.srt', 'v0394cg10000c6jlcn3c77u71r64sig0_68.srt', 'v0394cg10000c6jm09jc77ueqjdn13cg_69.srt', 'v0394cg10000c6jo3d3c77u4bvfqimu0_11.srt', 'v0394cg10000c6k3k13c77ufq1vudv80_35.srt', 'v0394cg10000c6k3k13c77ufq1vudv80_45.srt', 'v0394cg10000c6k3k13c77ufq1vudv80_49.srt', 'v0394cg10000c6k3k4jc77u9ofhav7cg_3.srt', 'v0394cg10000c6k3k4jc77u9ofhav7cg_33.srt', 'v0394cg10000c6k3k4jc77u9ofhav7cg_9.srt', 'v0394cg10000c6k8id3c77u1upvucajg_0.srt', 'v0394cg10000c6k8id3c77u1upvucajg_12.srt', 'v0394cg10000c6k916rc77u1j51ufn10_17.srt', 'v0394cg10000c6k916rc77u1j51ufn10_22.srt', 'v0394cg10000c6k916rc77u1j51ufn10_39.srt', 'v0394cg10000c6ksg7rc77u2i5u1a1c0_26.srt', 'v0394cg10000c6ksg7rc77u2i5u1a1c0_6.srt', 'v0394cg10000c6ksg7rc77u2i5u1a1c0_65.srt', 'v0394cg10000c6kt3r3c77u8d2e84gvg_18.srt', 'v0394cg10000c6ktc3rc77u44tsvbvfg_1.srt', 'v0394cg10000c6ktki3c77u1of66t5ug_32.srt', 'v0394cg10000c6ktki3c77u1of66t5ug_39.srt', 'v0394cg10000c6le0hrc77ualuo8ujfg_28.srt', 'v0394cg10000c6o4anbc77u1of18op60_19.srt', 'v0394cg10000c6o4anbc77u1of18op60_22.srt', 'v0394cg10000c6o4anbc77u1of18op60_34.srt', 'v0394cg10000c6oo41bc77u71j6iuvt0_26.srt', 'v0394cg10000c6oo41bc77u71j6iuvt0_38.srt', 'v0394cg10000c6oo41bc77u71j6iuvt0_42.srt', 'v0394cg10000c6oo41bc77u71j6iuvt0_46.srt', 'v0394cg10000c6ouuq3c77u40bem1m80_42.srt', 'v0394cg10000c6ouuq3c77u40bem1m80_48.srt', 'v0394cg10000c6ouuq3c77u40bem1m80_73.srt', 'v0394cg10000c6ouuq3c77u40bem1m80_81.srt', 'v0394cg10000c6sb7nbc77uekeep99tg_2.srt', 'v0394cg10000c6vl1arc77u03thmak7g_2.srt', 'v0394cg10000c6vl1arc77u03thmak7g_40.srt', 'v0394cg10000c73u4frc77ufmc1pi5d0_30.srt', 'v0394cg10000c73u4frc77ufmc1pi5d0_51.srt', 'v0394cg10000c74ssmbc77u283kt9tvg_18.srt', 'v0394cg10000c74ssmbc77u283kt9tvg_29.srt', 'v0394cg10000c74ssmbc77u283kt9tvg_33.srt', 'v0394cg10000c788pvbc77u0toro9r3g_12.srt', 'v0394cg10000c788pvbc77u0toro9r3g_23.srt', 'v0394cg10000c78n963c77u54k0r452g_25.srt', 'v0394cg10000c78n963c77u54k0r452g_6.srt', 'v0394cg10000c7f95i3c77u42ker8kdg_133.srt', 'v0394cg10000c7ptjebc77u3gr14g4c0_21.srt', 'v0394cg10000c7ptjebc77u3gr14g4c0_6.srt', 'v0394cg10000c7vtvv3c77ucbtus6uf0_47.srt', 'v0394cg10000c81t0ujc77u3d4hfliv0_32.srt', 'v0394cg10000c81t0ujc77u3d4hfliv0_8.srt', 'v0394cg10000c8aa41jc77u7h62v68u0_15.srt', 'v0394cg10000c8aa41jc77u7h62v68u0_31.srt', 'v0394cg10000c8aph53c77ue3t3qd5h0_71.srt', 'v0394cg10000c8aph53c77ue3t3qd5h0_80.srt', 'v0394cg10000c8fm47bc77u89tcsm6bg_5.srt', 'v0394cg10000c8fm47bc77u89tcsm6bg_54.srt', 'v0394cg10000c8g4s93c77udud6bthr0_53.srt', 'v0394cg10000c8g7ldrc77u8d5rvupig_68.srt', 'v0394cg10000c8gbg13c77u4pba3fuhg_61.srt', 'v0394cg10000c8gbg13c77u4pba3fuhg_72.srt', 'v0394cg10000c8gbj4jc77ua3s02v550_26.srt', 'v0394cg10000c8gbj4jc77ua3s02v550_35.srt', 'v0394cg10000c8gqi3jc77u4se2l29k0_9.srt', 'v0394cg10000c8gql53c77u1315opqd0_24.srt', 'v0394cg10000c8h5hbjc77u2iq7vmfjg_57.srt', 'v0394cg10000c9timejc77ueeu0u1r1g_35.srt', 'v0394cg10000cactq63c77u230lf7dkg_83.srt', 'v0394cg10000cae4la3c77ud8059lmdg_45.srt', 'v0394cg10000caffh33c77uddutsh6kg_7.srt', 'v0394cg10000cai9qjbc77uf67aig010_139.srt', 'v0394cg10000carruibc77u9i35or4h0_1.srt', 'v0394cg10000carruibc77u9i35or4h0_10.srt', 'v0394cg10000carruibc77u9i35or4h0_23.srt', 'v0394cg10000carruibc77u9i35or4h0_8.srt', 'v0394cg10000cb019ujc77u7t8a1b7tg_21.srt', 'v0394cg10000cb019ujc77u7t8a1b7tg_48.srt', 'v0d94cg10000c6jlii3c77udm6lvns1g_63.srt', 'v0d94cg10000c6jlii3c77udm6lvns1g_72.srt', 'v0d94cg10000c6jm28bc77u9q8i48c60_41.srt', 'v0d94cg10000c6kmlu3c77ub4alep4pg_34.srt', 'v0d94cg10000c6kmlu3c77ub4alep4pg_59.srt', 'v0d94cg10000c6kmlu3c77ub4alep4pg_63.srt', 'v0d94cg10000c6ko1krc77u8uphpp4fg_39.srt', 'v0d94cg10000c6kpk6jc77u20tbnr8hg_23.srt', 'v0d94cg10000c6kpk6jc77u20tbnr8hg_26.srt', 'v0d94cg10000c6kpk6jc77u20tbnr8hg_73.srt', 'v0d94cg10000c6kpk6jc77u20tbnr8hg_85.srt', 'v0d94cg10000c6kstv3c77u1v9c37b9g_17.srt', 'v0d94cg10000c6p1m53c77u9e6vm51r0_51.srt', 'v0d94cg10000c6p1m53c77u9e6vm51r0_53.srt', 'v0d94cg10000c74l1irc77u3rsiquvq0_9.srt', 'v0d94cg10000c7f3knbc77u72h67gf3g_11.srt', 'v0d94cg10000c7f3knbc77u72h67gf3g_49.srt', 'v0d94cg10000c7gv2sjc77ud8s41uv10_51.srt', 'v0d94cg10000c7gv2sjc77ud8s41uv10_65.srt', 'v0d94cg10000c7o6ft3c77u836u8akkg_20.srt', 'v0d94cg10000c7oipfbc77u6nnq0esa0_17.srt', 'v0d94cg10000c7oipfbc77u6nnq0esa0_47.srt', 'v0d94cg10000c7oipfbc77u6nnq0esa0_56.srt', 'v0d94cg10000c7rr8t3c77uclrlbh3t0_31.srt', 'v0d94cg10000c8bnugrc77ubq48jpiog_13.srt', 'v0d94cg10000c8bnugrc77ubq48jpiog_8.srt', 'v0d94cg10000c8fm53jc77u58ovpmd8g_4.srt', 'v0d94cg10000c8hcbj3c77u25j7or3jg_8.srt', 'v0d94cg10000c8hd17rc77u7bpore8g0_50.srt', 'v0d94cg10000c8hduurc77uc1purvr8g_11.srt', 'v0d94cg10000c8hduurc77uc1purvr8g_12.srt', 'v0d94cg10000c8hduurc77uc1purvr8g_22.srt', 'v0d94cg10000c8hduurc77uc1purvr8g_5.srt', 'v0d94cg10000c9h54nrc77uel579m7f0_81.srt', 'v0d94cg10000c9h54nrc77uel579m7f0_9.srt', 'v0d94cg10000c9qbsrrc77u9ef39kj10_25.srt', 'v0d94cg10000c9qbsrrc77u9ef39kj10_38.srt', 'v0d94cg10000c9u7fc3c77u1il8j9d2g_25.srt', 'v0d94cg10000ca49fu3c77u96g9lcu70_10.srt', 'v0d94cg10000ca9mcsrc77u9r5mkd2jg_14.srt', 'v0d94cg10000ca9mcsrc77u9r5mkd2jg_25.srt', 'v0d94cg10000caba61jc77u5v14qh1ng_61.srt', 'v0d94cg10000cae9oprc77u6mbu7d6p0_44.srt', 'v0d94cg10000cahk0jbc77u4k83e0uu0_19.srt', 'v0d94cg10000cahk0jbc77u4k83e0uu0_37.srt', 'v0d94cg10000cahk0jbc77u4k83e0uu0_88.srt', 'v0d94cg10000cavrebjc77ueu04f0k10_45.srt', 'v0d94cg10000cb02qmrc77u69j0su6ig_18.srt', 'v0d94cg10000cb02qmrc77u69j0su6ig_23.srt', 'v0d94cg10000cb02qmrc77u69j0su6ig_45.srt', 'v0d94cg10000cb02qmrc77u69j0su6ig_52.srt', 'v0d94cg10000cb02qmrc77u69j0su6ig_58.srt', 'v0d94cg10000cbb3adjc77ufrqg51la0_15.srt', 'v0d94cg10000cbb3adjc77ufrqg51la0_21.srt', 'v0d94cg10000cbb3adjc77ufrqg51la0_23.srt', 'v0d94cg10000cbb3adjc77ufrqg51la0_37.srt', 'v0d94cg10000cbb3adjc77ufrqg51la0_4.srt', 'vpEW7JZwE2E_04.srt', 'VZRyYGNrhR8_09.srt', 'WsfOorcJtEw_01.srt', 'WsfOorcJtEw_02.srt', 'WsfOorcJtEw_04.srt', 'WsfOorcJtEw_12.srt', 'x90oQ4cvYDc_07.srt', 'xEBaQypWAGA_39.srt', 'xEBaQypWAGA_68.srt', 'xfwi2WEbLbs_03.srt', 'xfwi2WEbLbs_06.srt', 'xfwi2WEbLbs_07.srt', 'xfwi2WEbLbs_27.srt', 'XJkb1rS0enI_122.srt', 'XJkb1rS0enI_42.srt', 'YcBVCMO97AE_07.srt', 'YcBVCMO97AE_16.srt', 'ZTO5wAdwj9A_24.srt']
    dirty_data_processor = DirtyDataProcessor("./evaluation/test_data/srt_output")
    dirty_data_processor.clean_dirty_data(dirty_data_list)
    
    
    # （如果你要测BigVideo数据集的话）然后就可以通过你刚才的得到的大量srt文件，来生成一个eval_result.zh文件了
    # 这个eval_result.zh文件会像是这样，每行对应数据集里一个视频的全部字幕（脏数据会被标记为。）
    """
    其次，我建议采用“润滑槽”方法，这意味着每天多次在单杠上悬挂，时间约为你最大悬挂时间的50%，这主要是进行次最大组的练习，你需要频繁练习，同时尽量保持身体的清新状态，每天都要进行“润滑槽”练习。
    它开始从你的脚下掉落，或者在顶级燃料车或前端车的情况下，它们有一个手刹杆用于制动，没有实际的刹车踏板，顶级燃料车甚至没有前刹车，只有后刹车。
    在光束重新连接时，他们基本上会有一点点的提前时间，然后计时器才会开始，顶级燃料车和搞笑车有外部启动器，他们会将其连接到增压器上。
    他们确实有一个叫做倒车器的部件，所以他们可以从烧胎中倒车。
    。
    但是，这里还有另一个与这个低压场的相互作用，可能比你预期的要更远。
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
