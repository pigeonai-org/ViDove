import sys
import numpy as np
sys.path.append('../src')
from srt_util.srt import SrtScript
from srt_util.srt import SrtSegment


"""
This file won't be used in the current evaluation module.    
"""


# Helper method
# Align sub anchor segment pair via greedy approach
# Input: anchor segment, SRT segments, output array of sub, index of current sub
# Output: updated index of sub
def procedure(anchor, subsec, S_arr, subidx):
    cache_idx = 0
    while subidx != cache_idx:  # Terminate when alignment stablizes
        cache_idx = subidx
        # if sub segment runs out during the loop, terminate
        if subidx >= len(subsec): 
            break
        sub = subsec[subidx]
        if anchor.end < sub.start:
            continue
        # If next sub has a heavier overlap compartment, add to current alignment
        if (anchor.start <= sub.start) and (sub.end <= anchor.end) or anchor.end - sub.start > sub.end - anchor.start:
            S_arr[-1] += sub#.source_text
            subidx += 1

    return subidx - 1  # Reset last invalid update from loop


# Input: path1, path2
# Output: aligned array of SRTsegment corresponding to path1 path2
# Note: Modify comment with .source_text to get output array with string only
def alignment_obsolete(pred_path, gt_path):
    empt = SrtSegment([0,'00:00:00,000 --> 00:00:00,000','','',''])
    pred = SrtScript.parse_from_srt_file(pred_path).segments
    gt = SrtScript.parse_from_srt_file(gt_path).segments
    pred_arr, gt_arr = [], []
    idx_p, idx_t = 0, 0  # idx_p: current index of pred segment, idx_t for ground truth

    while idx_p < len(pred) or idx_t < len(gt):
        # Check if one srt file runs out while reading
        ps = pred[idx_p] if idx_p < len(pred) else None
        gs = gt[idx_t] if idx_t < len(gt) else None
        
        if not ps:
            # If ps runs out, align gs segment with filler one by one
            gt_arr.append(gs)#.source_text
            pred_arr.append(empt)
            idx_t += 1
            continue

        if not gs:
            # If gs runs out, align ps segment with filler one by one
            pred_arr.append(ps)#.source_text 
            gt_arr.append(empt)
            idx_p += 1
            continue

        ps_dur = ps.end - ps.start
        gs_dur = gs.end - gs.start
        
        # Check for duration to decide anchor and sub
        if ps_dur <= gs_dur:
            # Detect segment with no overlap
            if ps.end < gs.start:
                pred_arr.append(ps)#.source_text
                gt_arr.append(empt)  # append filler
                idx_t -= 1  # reset ground truth index
            else:
                
                if gs.end >= ps.start:
                    gt_arr.append(gs)#.source_text
                    pred_arr.append(ps)#.source_text
                    idx_p = procedure(gs, pred, pred_arr, idx_p + 1)
                else:
                    gt_arr[len(gt_arr) - 1] += gs#.source_text
                    #pred_arr.append(empt)
                    idx_p -= 1
        else:
            # same overlap checking procedure
            if gs.end < ps.start:
                gt_arr.append(gs)#.source_text
                pred_arr.append(empt)  # filler
                idx_p -= 1  # reset
            else:
                if ps.end >= gs.start:
                    pred_arr.append(ps)#.source_text
                    gt_arr.append(gs)#.source_text
                    idx_t = procedure(ps, gt, gt_arr, idx_t + 1)
                else:  # filler pairing
                    pred_arr[len(pred_arr) - 1] += ps
                    idx_t -= 1

        idx_p += 1
        idx_t += 1
    #for a in gt_arr:
    #    print(a.translation)
    return zip(pred_arr, gt_arr)

# Input: path1, path2, threshold = 0.5 sec by default
# Output: aligned array of SRTsegment corresponding to path1 path2
def alignment(pred_path, gt_path, threshold=0.5):
    empt = SrtSegment([0, '00:00:00,000 --> 00:00:00,000', '', '', ''])
    pred = SrtScript.parse_from_srt_file(pred_path).segments
    gt = SrtScript.parse_from_srt_file(gt_path).segments
    pred_arr, gt_arr = [], []
    idx_p, idx_t = 0, 0

    while idx_p < len(pred) or idx_t < len(gt):
        ps = pred[idx_p] if idx_p < len(pred) else empt
        gs = gt[idx_t] if idx_t < len(gt) else empt

        # Merging sequence for pred
        while idx_p + 1 < len(pred) and pred[idx_p + 1].end <= gs.end + threshold:
            ps += pred[idx_p + 1]
            idx_p += 1

        # Merging sequence for gt
        while idx_t + 1 < len(gt) and gt[idx_t + 1].end <= ps.end + threshold:
            gs += gt[idx_t + 1]
            idx_t += 1

        # Append to the result arrays
        pred_arr.append(ps)
        gt_arr.append(gs)
        idx_p += 1
        idx_t += 1


    #for a in pred_arr:
    #    print(a.translation)
    #for a in gt_arr:
    #    print(a.source_text)

    return zip(pred_arr, gt_arr)


#  Test Case
#alignment('test_translation_s2.srt', 'test_translation_zh.srt')
