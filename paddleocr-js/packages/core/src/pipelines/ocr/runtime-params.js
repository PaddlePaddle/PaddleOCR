function firstDefined(...values) {
  for (const value of values) {
    if (value !== undefined && value !== null) {
      return value;
    }
  }
  return undefined;
}

export function getOcrRuntimeParams(config, defaults = {}, params = {}) {
  return {
    text_det_limit_side_len: Number(
      firstDefined(
        params.text_det_limit_side_len,
        params.textDetLimitSideLen,
        defaults.text_det_limit_side_len,
        config.det.resizeLong
      )
    ),
    text_det_limit_type:
      firstDefined(
        params.text_det_limit_type,
        params.textDetLimitType,
        defaults.text_det_limit_type
      ) || "max",
    text_det_max_side_limit: Number(
      firstDefined(
        params.text_det_max_side_limit,
        params.textDetMaxSideLimit,
        defaults.text_det_max_side_limit,
        config.det.maxSideLimit
      )
    ),
    text_det_thresh: Number(
      firstDefined(
        params.text_det_thresh,
        params.textDetThresh,
        defaults.text_det_thresh,
        config.det.postprocess.thresh
      )
    ),
    text_det_box_thresh: Number(
      firstDefined(
        params.text_det_box_thresh,
        params.textDetBoxThresh,
        defaults.text_det_box_thresh,
        config.det.postprocess.boxThresh
      )
    ),
    text_det_unclip_ratio: Number(
      firstDefined(
        params.text_det_unclip_ratio,
        params.textDetUnclipRatio,
        defaults.text_det_unclip_ratio,
        config.det.postprocess.unclipRatio
      )
    ),
    text_rec_score_thresh: Number(
      firstDefined(
        params.text_rec_score_thresh,
        params.textRecScoreThresh,
        defaults.text_rec_score_thresh,
        config.rec.scoreThresh
      )
    )
  };
}
