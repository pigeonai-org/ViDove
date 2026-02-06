# identify dirty data base on evaluation/test_data/eval_result.zh and evaluation/test_data/test_data_test.id

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# temporarily hard code
dirty_data_list = ['00MkZbEwoZE_14.srt', '00MkZbEwoZE_20.srt', '0Tkvi_YDlK4_00.srt', '0Tkvi_YDlK4_15.srt', '0UakYLTiplc_00.srt', '0UakYLTiplc_19.srt', '0UakYLTiplc_21.srt', '0UakYLTiplc_24.srt', '3t0IlJtGO88_27.srt', '6a1MWu1MdvE_11.srt', '7f1s1ubpHF4_06.srt', '7f1s1ubpHF4_12.srt', '7f1s1ubpHF4_13.srt', '7f1s1ubpHF4_14.srt', '7taB6hQ7430_38.srt', '8xF_Dfe4rn8_00.srt', '8xF_Dfe4rn8_107.srt', '8xF_Dfe4rn8_137.srt', '8xF_Dfe4rn8_138.srt', '8xF_Dfe4rn8_38.srt', '8xF_Dfe4rn8_46.srt', '8xF_Dfe4rn8_72.srt', '8xF_Dfe4rn8_77.srt', '9zwO6I96oJg_01.srt', '9zwO6I96oJg_24.srt', 'aUN5EdBHHtA_24.srt', 'BbkvWitFkns_48.srt', 'ChnMBUVLhgA_05.srt', 'ChnMBUVLhgA_20.srt', 'CsXGMeMc75U_06.srt', 'CXKVmQYGDu4_12.srt', 'CYz8zVAMpfw_04.srt', 'CYz8zVAMpfw_19.srt', 'D4a_i3rpLzg_01.srt', 'Do51eJowSyg_102.srt', 'Do51eJowSyg_105.srt', 'Do51eJowSyg_113.srt', 'Do51eJowSyg_138.srt', 'Do51eJowSyg_141.srt', 'Do51eJowSyg_144.srt', 'Do51eJowSyg_41.srt', 'Do51eJowSyg_45.srt', 'Do51eJowSyg_58.srt', 'Do51eJowSyg_59.srt', 'Do51eJowSyg_68.srt', 'Do51eJowSyg_86.srt', 'Do51eJowSyg_87.srt', 'Do51eJowSyg_95.srt', 'Do51eJowSyg_98.srt', 'f4KOKfhqp7w_06.srt', 'f4KOKfhqp7w_11.srt', 'GHWxQ9Bphm8_00.srt', 'GM9jTRFCtj8_00.srt', 'hWpbco3F4L4_107.srt', 'hWpbco3F4L4_115.srt', 'hWpbco3F4L4_157.srt', 'hWpbco3F4L4_158.srt', 'hWpbco3F4L4_21.srt', 'hWpbco3F4L4_24.srt', 'hWpbco3F4L4_25.srt', 'IT71qzZw6B8_01.srt', 'Jx2kXT2o_Zw_23.srt', 'mZqpsnPCSIw_10.srt', 'mZqpsnPCSIw_28.srt', 'NEFVpQ66KEo_02.srt', 'NEFVpQ66KEo_29.srt', 'NtKyiGbD6_4_03.srt', 'NtKyiGbD6_4_08.srt', 'NtKyiGbD6_4_16.srt', 'NtKyiGbD6_4_27.srt', 'NtKyiGbD6_4_34.srt', 'QA2HuYGCALQ_09.srt', 'qOzpF9LUw64_05.srt', 'SYjRgxNRDOY_02.srt', 'SYjRgxNRDOY_26.srt', 'SYjRgxNRDOY_27.srt', 'SYjRgxNRDOY_35.srt', 'SYjRgxNRDOY_42.srt', 'SYjRgxNRDOY_45.srt', 'SYjRgxNRDOY_47.srt', 'SYjRgxNRDOY_60.srt', 'SYjRgxNRDOY_73.srt', 'SYjRgxNRDOY_74.srt', 'SYjRgxNRDOY_79.srt', 'UoStHRC43H8_07.srt', 'UoStHRC43H8_16.srt', 'v0294cg10000c6hohc3c77u95on42b7g_171.srt', 'v0294cg10000c6hohc3c77u95on42b7g_376.srt', 'v0294cg10000c6hohc3c77u95on42b7g_49.srt', 'v0294cg10000c6j2qk3c77u8onkj7ihg_161.srt', 'v0294cg10000c6jf36rc77u94potnq90_43.srt', 'v0294cg10000c6jq6ajc77u5ga5iu8c0_121.srt', 'v0294cg10000c6jq6ajc77u5ga5iu8c0_26.srt', 'v0294cg10000c6jq6ajc77u5ga5iu8c0_89.srt', 'v0294cg10000c6k4td3c77u77qrjmbf0_0.srt', 'v0294cg10000c6kt6l3c77ucaaar4kr0_33.srt', 'v0294cg10000c6kt6l3c77ucaaar4kr0_58.srt', 'v0294cg10000c6le7obc77u62sv6k3h0_0.srt', 'v0294cg10000c6le7obc77u62sv6k3h0_39.srt', 'v0294cg10000c6le7obc77u62sv6k3h0_4.srt', 'v0294cg10000c6le7obc77u62sv6k3h0_41.srt', 'v0294cg10000c6le7obc77u62sv6k3h0_8.srt', 'v0294cg10000c6t0j2rc77u6rv3b0p0g_23.srt', 'v0294cg10000c6t0j2rc77u6rv3b0p0g_29.srt', 'v0294cg10000c6t0j2rc77u6rv3b0p0g_30.srt', 'v0294cg10000c74odf3c77uagidi1qf0_2.srt', 'v0294cg10000c7iqslbc77u2p0o1ea80_147.srt', 'v0294cg10000c7rd73rc77u0ehls07kg_39.srt', 'v0294cg10000c7rujq3c77u2vf6ibo60_17.srt', 'v0294cg10000c83si03c77ue1vchkdj0_29.srt', 'v0294cg10000c83si03c77ue1vchkdj0_50.srt', 'v0294cg10000c83si03c77ue1vchkdj0_67.srt', 'v0294cg10000c83si03c77ue1vchkdj0_70.srt', 'v0294cg10000c874u4bc77u5f28dm590_39.srt', 'v0294cg10000c874u4bc77u5f28dm590_96.srt', 'v0294cg10000c87l78jc77u52ggje78g_12.srt', 'v0294cg10000c87l78jc77u52ggje78g_16.srt', 'v0294cg10000c87l78jc77u52ggje78g_21.srt', 'v0294cg10000c87l78jc77u52ggje78g_23.srt', 'v0294cg10000c87l78jc77u52ggje78g_32.srt', 'v0294cg10000c89694jc77u0850rohl0_49.srt', 'v0294cg10000c8rct33c77udjr3hdlug_28.srt', 'v0294cg10000c8rct33c77udjr3hdlug_36.srt', 'v0294cg10000c9cfuprc77ua1qpu2qgg_25.srt', 'v0294cg10000c9igmebc77u5bcl3vtk0_2.srt', 'v0294cg10000c9igmebc77u5bcl3vtk0_29.srt', 'v0294cg10000c9r0sibc77ub3rvsiuj0_3.srt', 'v0294cg10000c9sedlrc77ue02r00pog_38.srt', 'v0294cg10000c9u7kljc77u9hrcmsukg_38.srt', 'v0294cg10000c9u7kljc77u9hrcmsukg_82.srt', 'v0294cg10000cacn6ujc77u3c6j03o70_1.srt', 'v0294cg10000camip7jc77uddhph5qm0_44.srt', 'v0294cg10000cb8j90jc77u338nnc6s0_52.srt', 'v0294cg10000cbb406rc77udk5e968i0_15.srt', 'v0294cg10000cbb406rc77udk5e968i0_31.srt', 'v0394cg10000c40fno3c77u62s93ugrg_1.srt', 'v0394cg10000c40fno3c77u62s93ugrg_2.srt', 'v0394cg10000c40fno3c77u62s93ugrg_23.srt', 'v0394cg10000c40fno3c77u62s93ugrg_3.srt', 'v0394cg10000c40fno3c77u62s93ugrg_32.srt', 'v0394cg10000c6h51erc77u20sog1hgg_25.srt', 'v0394cg10000c6i48ubc77u43t50pgsg_14.srt', 'v0394cg10000c6i48ubc77u43t50pgsg_15.srt', 'v0394cg10000c6i48ubc77u43t50pgsg_22.srt', 'v0394cg10000c6i48ubc77u43t50pgsg_5.srt', 'v0394cg10000c6i48ubc77u43t50pgsg_79.srt', 'v0394cg10000c6iar1rc77ucsqln0brg_146.srt', 'v0394cg10000c6it0crc77u40bd5qfrg_28.srt', 'v0394cg10000c6ji6njc77u8bvf1hb00_1.srt', 'v0394cg10000c6ji6njc77u8bvf1hb00_10.srt', 'v0394cg10000c6ji6njc77u8bvf1hb00_53.srt', 'v0394cg10000c6ji6njc77u8bvf1hb00_6.srt', 'v0394cg10000c6ji6njc77u8bvf1hb00_68.srt', 'v0394cg10000c6ji6njc77u8bvf1hb00_70.srt', 'v0394cg10000c6jjjtrc77u8bkp0uu60_3.srt', 'v0394cg10000c6jjjtrc77u8bkp0uu60_8.srt', 'v0394cg10000c6jjmojc77u03tn4qlpg_140.srt', 'v0394cg10000c6jjmojc77u03tn4qlpg_29.srt', 'v0394cg10000c6jjmojc77u03tn4qlpg_63.srt', 'v0394cg10000c6jlcn3c77u71r64sig0_62.srt', 'v0394cg10000c6jlcn3c77u71r64sig0_68.srt', 'v0394cg10000c6jm09jc77ueqjdn13cg_69.srt', 'v0394cg10000c6jo3d3c77u4bvfqimu0_11.srt', 'v0394cg10000c6k3k13c77ufq1vudv80_35.srt', 'v0394cg10000c6k3k13c77ufq1vudv80_45.srt', 'v0394cg10000c6k3k13c77ufq1vudv80_49.srt', 'v0394cg10000c6k3k4jc77u9ofhav7cg_3.srt', 'v0394cg10000c6k3k4jc77u9ofhav7cg_33.srt', 'v0394cg10000c6k3k4jc77u9ofhav7cg_9.srt', 'v0394cg10000c6k8id3c77u1upvucajg_0.srt', 'v0394cg10000c6k8id3c77u1upvucajg_12.srt', 'v0394cg10000c6k916rc77u1j51ufn10_17.srt', 'v0394cg10000c6k916rc77u1j51ufn10_22.srt', 'v0394cg10000c6k916rc77u1j51ufn10_39.srt', 'v0394cg10000c6ksg7rc77u2i5u1a1c0_26.srt', 'v0394cg10000c6ksg7rc77u2i5u1a1c0_6.srt', 'v0394cg10000c6ksg7rc77u2i5u1a1c0_65.srt', 'v0394cg10000c6kt3r3c77u8d2e84gvg_18.srt', 'v0394cg10000c6ktc3rc77u44tsvbvfg_1.srt', 'v0394cg10000c6ktki3c77u1of66t5ug_32.srt', 'v0394cg10000c6ktki3c77u1of66t5ug_39.srt', 'v0394cg10000c6le0hrc77ualuo8ujfg_28.srt', 'v0394cg10000c6o4anbc77u1of18op60_19.srt', 'v0394cg10000c6o4anbc77u1of18op60_22.srt', 'v0394cg10000c6o4anbc77u1of18op60_34.srt', 'v0394cg10000c6oo41bc77u71j6iuvt0_26.srt', 'v0394cg10000c6oo41bc77u71j6iuvt0_38.srt', 'v0394cg10000c6oo41bc77u71j6iuvt0_42.srt', 'v0394cg10000c6oo41bc77u71j6iuvt0_46.srt', 'v0394cg10000c6ouuq3c77u40bem1m80_42.srt', 'v0394cg10000c6ouuq3c77u40bem1m80_48.srt', 'v0394cg10000c6ouuq3c77u40bem1m80_73.srt', 'v0394cg10000c6ouuq3c77u40bem1m80_81.srt', 'v0394cg10000c6sb7nbc77uekeep99tg_2.srt', 'v0394cg10000c6vl1arc77u03thmak7g_2.srt', 'v0394cg10000c6vl1arc77u03thmak7g_40.srt', 'v0394cg10000c73u4frc77ufmc1pi5d0_30.srt', 'v0394cg10000c73u4frc77ufmc1pi5d0_51.srt', 'v0394cg10000c74ssmbc77u283kt9tvg_18.srt', 'v0394cg10000c74ssmbc77u283kt9tvg_29.srt', 'v0394cg10000c74ssmbc77u283kt9tvg_33.srt', 'v0394cg10000c788pvbc77u0toro9r3g_12.srt', 'v0394cg10000c788pvbc77u0toro9r3g_23.srt', 'v0394cg10000c78n963c77u54k0r452g_25.srt', 'v0394cg10000c78n963c77u54k0r452g_6.srt', 'v0394cg10000c7f95i3c77u42ker8kdg_133.srt', 'v0394cg10000c7ptjebc77u3gr14g4c0_21.srt', 'v0394cg10000c7ptjebc77u3gr14g4c0_6.srt', 'v0394cg10000c7vtvv3c77ucbtus6uf0_47.srt', 'v0394cg10000c81t0ujc77u3d4hfliv0_32.srt', 'v0394cg10000c81t0ujc77u3d4hfliv0_8.srt', 'v0394cg10000c8aa41jc77u7h62v68u0_15.srt', 'v0394cg10000c8aa41jc77u7h62v68u0_31.srt', 'v0394cg10000c8aph53c77ue3t3qd5h0_71.srt', 'v0394cg10000c8aph53c77ue3t3qd5h0_80.srt', 'v0394cg10000c8fm47bc77u89tcsm6bg_5.srt', 'v0394cg10000c8fm47bc77u89tcsm6bg_54.srt', 'v0394cg10000c8g4s93c77udud6bthr0_53.srt', 'v0394cg10000c8g7ldrc77u8d5rvupig_68.srt', 'v0394cg10000c8gbg13c77u4pba3fuhg_61.srt', 'v0394cg10000c8gbg13c77u4pba3fuhg_72.srt', 'v0394cg10000c8gbj4jc77ua3s02v550_26.srt', 'v0394cg10000c8gbj4jc77ua3s02v550_35.srt', 'v0394cg10000c8gqi3jc77u4se2l29k0_9.srt', 'v0394cg10000c8gql53c77u1315opqd0_24.srt', 'v0394cg10000c8h5hbjc77u2iq7vmfjg_57.srt', 'v0394cg10000c9timejc77ueeu0u1r1g_35.srt', 'v0394cg10000cactq63c77u230lf7dkg_83.srt', 'v0394cg10000cae4la3c77ud8059lmdg_45.srt', 'v0394cg10000caffh33c77uddutsh6kg_7.srt', 'v0394cg10000cai9qjbc77uf67aig010_139.srt', 'v0394cg10000carruibc77u9i35or4h0_1.srt', 'v0394cg10000carruibc77u9i35or4h0_10.srt', 'v0394cg10000carruibc77u9i35or4h0_23.srt', 'v0394cg10000carruibc77u9i35or4h0_8.srt', 'v0394cg10000cb019ujc77u7t8a1b7tg_21.srt', 'v0394cg10000cb019ujc77u7t8a1b7tg_48.srt', 'v0d94cg10000c6jlii3c77udm6lvns1g_63.srt', 'v0d94cg10000c6jlii3c77udm6lvns1g_72.srt', 'v0d94cg10000c6jm28bc77u9q8i48c60_41.srt', 'v0d94cg10000c6kmlu3c77ub4alep4pg_34.srt', 'v0d94cg10000c6kmlu3c77ub4alep4pg_59.srt', 'v0d94cg10000c6kmlu3c77ub4alep4pg_63.srt', 'v0d94cg10000c6ko1krc77u8uphpp4fg_39.srt', 'v0d94cg10000c6kpk6jc77u20tbnr8hg_23.srt', 'v0d94cg10000c6kpk6jc77u20tbnr8hg_26.srt', 'v0d94cg10000c6kpk6jc77u20tbnr8hg_73.srt', 'v0d94cg10000c6kpk6jc77u20tbnr8hg_85.srt', 'v0d94cg10000c6kstv3c77u1v9c37b9g_17.srt', 'v0d94cg10000c6p1m53c77u9e6vm51r0_51.srt', 'v0d94cg10000c6p1m53c77u9e6vm51r0_53.srt', 'v0d94cg10000c74l1irc77u3rsiquvq0_9.srt', 'v0d94cg10000c7f3knbc77u72h67gf3g_11.srt', 'v0d94cg10000c7f3knbc77u72h67gf3g_49.srt', 'v0d94cg10000c7gv2sjc77ud8s41uv10_51.srt', 'v0d94cg10000c7gv2sjc77ud8s41uv10_65.srt', 'v0d94cg10000c7o6ft3c77u836u8akkg_20.srt', 'v0d94cg10000c7oipfbc77u6nnq0esa0_17.srt', 'v0d94cg10000c7oipfbc77u6nnq0esa0_47.srt', 'v0d94cg10000c7oipfbc77u6nnq0esa0_56.srt', 'v0d94cg10000c7rr8t3c77uclrlbh3t0_31.srt', 'v0d94cg10000c8bnugrc77ubq48jpiog_13.srt', 'v0d94cg10000c8bnugrc77ubq48jpiog_8.srt', 'v0d94cg10000c8fm53jc77u58ovpmd8g_4.srt', 'v0d94cg10000c8hcbj3c77u25j7or3jg_8.srt', 'v0d94cg10000c8hd17rc77u7bpore8g0_50.srt', 'v0d94cg10000c8hduurc77uc1purvr8g_11.srt', 'v0d94cg10000c8hduurc77uc1purvr8g_12.srt', 'v0d94cg10000c8hduurc77uc1purvr8g_22.srt', 'v0d94cg10000c8hduurc77uc1purvr8g_5.srt', 'v0d94cg10000c9h54nrc77uel579m7f0_81.srt', 'v0d94cg10000c9h54nrc77uel579m7f0_9.srt', 'v0d94cg10000c9qbsrrc77u9ef39kj10_25.srt', 'v0d94cg10000c9qbsrrc77u9ef39kj10_38.srt', 'v0d94cg10000c9u7fc3c77u1il8j9d2g_25.srt', 'v0d94cg10000ca49fu3c77u96g9lcu70_10.srt', 'v0d94cg10000ca9mcsrc77u9r5mkd2jg_14.srt', 'v0d94cg10000ca9mcsrc77u9r5mkd2jg_25.srt', 'v0d94cg10000caba61jc77u5v14qh1ng_61.srt', 'v0d94cg10000cae9oprc77u6mbu7d6p0_44.srt', 'v0d94cg10000cahk0jbc77u4k83e0uu0_19.srt', 'v0d94cg10000cahk0jbc77u4k83e0uu0_37.srt', 'v0d94cg10000cahk0jbc77u4k83e0uu0_88.srt', 'v0d94cg10000cavrebjc77ueu04f0k10_45.srt', 'v0d94cg10000cb02qmrc77u69j0su6ig_18.srt', 'v0d94cg10000cb02qmrc77u69j0su6ig_23.srt', 'v0d94cg10000cb02qmrc77u69j0su6ig_45.srt', 'v0d94cg10000cb02qmrc77u69j0su6ig_52.srt', 'v0d94cg10000cb02qmrc77u69j0su6ig_58.srt', 'v0d94cg10000cbb3adjc77ufrqg51la0_15.srt', 'v0d94cg10000cbb3adjc77ufrqg51la0_21.srt', 'v0d94cg10000cbb3adjc77ufrqg51la0_23.srt', 'v0d94cg10000cbb3adjc77ufrqg51la0_37.srt', 'v0d94cg10000cbb3adjc77ufrqg51la0_4.srt', 'vpEW7JZwE2E_04.srt', 'VZRyYGNrhR8_09.srt', 'WsfOorcJtEw_01.srt', 'WsfOorcJtEw_02.srt', 'WsfOorcJtEw_04.srt', 'WsfOorcJtEw_12.srt', 'x90oQ4cvYDc_07.srt', 'xEBaQypWAGA_39.srt', 'xEBaQypWAGA_68.srt', 'xfwi2WEbLbs_03.srt', 'xfwi2WEbLbs_06.srt', 'xfwi2WEbLbs_07.srt', 'xfwi2WEbLbs_27.srt', 'XJkb1rS0enI_122.srt', 'XJkb1rS0enI_42.srt', 'YcBVCMO97AE_07.srt', 'YcBVCMO97AE_16.srt', 'ZTO5wAdwj9A_24.srt']
    
class DirtyDataProcessor:
    def __init__(self, srt_result_path):
       self.srt_result_path = srt_result_path

    # 这个path下有很多srt文件，需要读取每个srt文件，如果有srt文件内容什么都没有的话，那么就认为这个srt文件是脏数据，他的文件名需要记录下来
    def identify_dirty_data(self):
        dirty_data_list = []
        for file in os.listdir(self.srt_result_path):
            if file.endswith(".srt"):
                with open(os.path.join(self.srt_result_path, file), "r", encoding="utf-8") as f:
                    if f.read() == "":
                        dirty_data_list.append(file)
                    # else:
                    #     print(f.read())
        print(len(dirty_data_list))
        return dirty_data_list
    
    def clean_dirty_data(self,dirty_data_list):
        cleaned_dirty_data = 0
        # dirty_data_list = self.identify_dirty_data()
        for file in dirty_data_list:
            # 写入这个文件，文件内容为""
            with open(os.path.join(self.srt_result_path, file), "w", encoding="utf-8") as f:
                f.write("")
                print("clean")
                cleaned_dirty_data += 1
        print(f"clean dirty data:{cleaned_dirty_data}")
    def add_period_to_empty_lines(self,file):
        # 遍历目录下的所有srt文件
        file = Path(file)
        # 读取文件内容
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # 处理空行
        new_lines = []
        for line in lines:
            if line.strip() == "":
                new_lines.append("。\n")
            else:
                new_lines.append(line)
        
        # 写回文件
        with open(file, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print("successfully done")

if __name__ == "__main__":
    dirty_data_processor = DirtyDataProcessor("./evaluation/test_data/gemini_results")
    
    # 生成一个dirty_data_list
    # print(dirty_data_processor.identify_dirty_data())
    # print(len(dirty_data_list))
    
    
    # 把所有dirty data给变成无内容的文件
    # dirty_data_processor.clean_dirty_data(dirty_data_list)
    
    # 给所有空行加上一个句号
    dirty_data_processor.add_period_to_empty_lines("evaluation/test_data/gemini_eval_result.zh")