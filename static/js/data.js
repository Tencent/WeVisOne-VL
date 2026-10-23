/* Real WeVisOne-VL-4B outputs (see outputs/ in the repo) and benchmark numbers from the technical report. */

window.DEMOS = [
  {
    id: "detection",
    label: "Detection",
    task: "detection",
    image: "static/images/demo_dogs.webp",
    prompt: "Locate all the instances that match the following description: dog. chair. bush.",
    response: "<ref>dog</ref><box><13><274><399><949>,<379><222><620><954>,<581><489><996><997>,<637><61><834><539></box>, <ref>chair</ref><box><416><0><586><208>,<708><0><832><139></box>, <ref>bush</ref><box><0><0><327><259></box>"
  },
  {
    id: "referring",
    label: "Referring",
    task: "referring",
    image: "static/images/demo_dogs.webp",
    prompt: "Locate the instances that match the following description: the black and white dog.",
    response: "<ref>the black and white dog</ref><box><13><274><398><948></box>"
  },
  {
    id: "count",
    label: "Counting",
    task: "count_box",
    image: "static/images/demo_dogs.webp",
    prompt: "How many dogs in the image? Please count them with boxes.",
    response: "<ref>dogs</ref><box><12><272><400><949>,<378><221><620><954>,<580><487><998><999>,<637><60><834><541></box>\nTotal number: 4."
  },
  {
    id: "point",
    label: "Pointing",
    task: "point",
    image: "static/images/demo_dogs.webp",
    prompt: "point to: dogs",
    response: "<ref>dogs</ref><point><204><612>,<500><587>,<788><726>,<739><295></point>"
  },
  {
    id: "ocr",
    label: "OCR",
    task: "ocr",
    image: "static/images/demo_coffee.webp",
    prompt: "detect all the word in the box format",
    response: "<ref>COFFEE</ref><box><450><33><850><250></box>, <ref>REAL</ref><box><94><62><406><245></box>, <ref>&</ref><box><443><116><540><179></box>, <ref>Tea</ref><box><308><184><452><242></box>, <ref>Café</ref><box><470><187><666><252></box>, <ref>10</ref><box><107><289><193><342></box>, <ref>METERS</ref><box><251><291><751><357></box>, <ref>AHEAD</ref><box><296><363><647><423></box>, <ref>TURN</ref><box><135><422><410><487></box>, <ref>RIGHT</ref><box><457><429><775><491></box>, <ref>IN</ref><box><424><492><515><547></box>, <ref>SMALL</ref><box><129><544><480><606></box>, <ref>ALLEY</ref><box><512><548><799><609></box>"
  },
  {
    id: "ground",
    label: "Text Grounding",
    task: "ocr_grounding",
    image: "static/images/demo_coffee.webp",
    prompt: "Locate the following text in the image: SMALL ALLEY",
    response: "<ref>SMALL ALLEY</ref><box><122><536><804><614></box>"
  },
  {
    id: "matting",
    label: "Matting",
    task: "matting",
    image: "static/images/demo_seabreeze.webp",
    matte: "static/images/demo_seabreeze_cutout.webp",
    prompt: "Locate all the instances that match the following description and provide the alpha mattes: person.",
    response: "<ref>person</ref><box><9><65><752><1000><OBJ></box>"
  }
];

/* Model groups drive bar colors: ours | vlm (grounding VLMs) | gen (general VLMs) | spec (specialists). */
(function () {
  const OURS4 = ["WeVisOne-VL-4B", "ours"];
  const OURS2 = ["WeVisOne-VL-2B", "ours"];
  const OBJ_MODELS = [
    ["Qwen3-VL-2B", "gen"], ["Qwen3-VL-4B", "gen"], ["Qwen3-VL-8B", "gen"],
    ["Qwen3.5-4B", "gen"], ["Qwen3.5-9B", "gen"],
    ["Rex-Omni-3B", "vlm"], ["LocateAnything-3B", "vlm"], OURS2, OURS4
  ];

  function table(models, columns, rows) {
    return columns.map((col, ci) => ({
      name: col.name,
      metric: col.metric,
      lower: !!col.lower,
      rows: models
        .map((m, mi) => ({ model: m[0], group: m[1], value: rows[mi][ci] }))
        .filter((r) => r.value !== null && r.value !== undefined)
    }));
  }

  const det = table(
    [["GroundingDINO-T", "spec"]].concat(OBJ_MODELS),
    [
      { name: "COCO", metric: "F1 mean (IoU 0.5:0.95)" },
      { name: "LVIS", metric: "F1 mean (IoU 0.5:0.95)" },
      { name: "Dense200", metric: "F1 mean (IoU 0.5:0.95)" },
      { name: "VisDrone", metric: "F1 mean (IoU 0.5:0.95)" }
    ],
    [
      [56.6, 38.8, 33.1, 38.5],
      [45.2, 40.8, 10.0, 23.1],
      [48.7, 44.0, 15.2, 31.0],
      [47.8, 42.9, 13.8, 29.5],
      [48.2, 42.2, 18.9, 29.9],
      [49.5, 43.7, 18.0, 31.3],
      [52.9, 46.9, 58.3, 35.8],
      [54.7, 50.7, 58.7, 39.9],
      [55.2, 53.3, 61.7, 40.4],
      [57.1, 55.3, 71.3, 44.7]
    ]
  );

  const ref = table(
    [["GroundingDINO-T", "spec"]].concat(OBJ_MODELS),
    [
      { name: "RefCOCOg val", metric: "F1 mean" },
      { name: "RefCOCOg-Rej", metric: "Rejection rate (%)" },
      { name: "WeRef-Bench", metric: "F1 mean" },
      { name: "WeRef-Bench Rej", metric: "Rejection rate (%)" },
      { name: "PR-Bench", metric: "F1 mean" },
      { name: "PR-Bench Rej", metric: "Rejection rate (%)" },
      { name: "HumanRef", metric: "F1 mean" },
      { name: "Ref-Adv", metric: "F1 mean (Ref-Adv-s)" },
      { name: "DVGBench", metric: "F1 mean" },
      { name: "Ref-L4", metric: "F1 mean" }
    ],
    [
      [45.9, 0.0, 21.1, 14.7, 25.2, 0.0, 25.2, 19.7, 24.1, 23.1],
      [70.8, 1.7, 30.8, 22.8, 41.6, 4.3, 63.2, 34.7, 35.2, 67.9],
      [74.1, 18.8, 46.3, 61.9, 49.9, 10.8, 69.4, 44.4, 42.4, 72.5],
      [74.6, 22.5, 36.7, 18.3, 50.3, 7.1, 71.6, 45.0, 43.4, 72.4],
      [73.3, 18.9, 48.1, 71.6, 49.6, 8.7, 75.8, 44.2, 40.2, 70.4],
      [74.3, 15.2, 50.7, 70.7, 50.8, 8.3, 77.8, 47.3, 42.5, 72.0],
      [73.6, 0.5, 50.3, 43.3, 42.7, 0.2, 79.9, 40.9, 33.2, 65.2],
      [76.7, 0.6, 51.7, 73.0, 44.5, 0.2, 78.7, 43.5, 34.7, 74.3],
      [77.2, 50.2, 73.2, 92.9, 50.0, 37.0, 77.3, 43.6, 40.7, 75.0],
      [80.8, 67.4, 78.7, 96.4, 54.1, 52.6, 80.7, 50.5, 45.2, 78.1]
    ]
  );

  const cnt = table(
    [["GroundingDINO-T", "spec"]].concat(OBJ_MODELS),
    [
      { name: "CountQA", metric: "Exact-match accuracy (%)" },
      { name: "PixMo-Count val", metric: "Exact-match accuracy (%)" },
      { name: "PixMo-Count test", metric: "Exact-match accuracy (%)" },
      { name: "CLOC val", metric: "Exact-match accuracy (%)" },
      { name: "CLOC test", metric: "Exact-match accuracy (%)" },
      { name: "KubriCount A", metric: "Exact-match accuracy (%)" },
      { name: "KubriCount B", metric: "Exact-match accuracy (%)" },
      { name: "MixCount val", metric: "Exact-match accuracy (%)" },
      { name: "MixCount test", metric: "Exact-match accuracy (%)" }
    ],
    [
      [9.6, 44.8, 48.3, 12.4, 9.5, 5.2, 6.5, 1.7, 2.8],
      [20.4, 61.3, 60.4, 16.3, 15.3, 4.3, 7.8, 1.8, 2.7],
      [29.2, 69.6, 65.6, 19.7, 18.6, 7.1, 12.1, 3.5, 4.3],
      [22.4, 68.2, 59.4, 18.5, 17.2, 6.6, 10.8, 3.2, 3.5],
      [40.0, 80.6, 76.9, 21.1, 21.7, 10.0, 16.2, 3.9, 5.2],
      [41.6, 82.6, 80.4, 23.1, 22.4, 11.9, 19.1, 5.8, 7.0],
      [37.6, 82.2, 71.9, 19.0, 22.3, 13.5, 18.8, 9.9, 11.4],
      [52.4, 88.7, 82.8, 22.8, 23.9, 18.5, 22.4, 12.5, 15.2],
      [47.9, 90.9, 88.1, 27.5, 26.9, 29.5, 32.4, 30.1, 32.2],
      [57.1, 93.7, 89.6, 28.2, 28.3, 37.7, 40.1, 42.0, 40.4]
    ]
  );

  const pt = table(
    OBJ_MODELS,
    [
      { name: "PointBench", metric: "F1@Point" },
      { name: "PixMo-Points", metric: "F1@Point" },
      { name: "COCO", metric: "F1@Point" },
      { name: "LVIS", metric: "F1@Point" },
      { name: "Dense200", metric: "F1@Point" },
      { name: "VisDrone", metric: "F1@Point" }
    ],
    [
      [61.2, 62.1, 72.7, 61.9, 29.5, 35.7],
      [57.1, 71.3, 74.5, 65.8, 37.7, 45.6],
      [62.5, 72.2, 77.5, 67.7, 42.2, 46.4],
      [61.5, 59.2, 72.7, 62.4, 7.4, 38.3],
      [66.4, 62.6, 74.4, 64.8, 10.5, 40.6],
      [69.9, 72.9, 80.5, 70.8, 82.5, 58.9],
      [56.9, 68.6, 83.9, 76.6, 87.6, 60.4],
      [62.9, 73.2, 85.8, 78.9, 83.2, 62.4],
      [71.9, 74.7, 85.4, 79.5, 87.0, 64.4]
    ]
  );

  const ocr = table(
    [["PP-OCRv5", "spec"]].concat(OBJ_MODELS),
    [
      { name: "HierText", metric: "F1 mean (IoU 0.5:0.95)" },
      { name: "ICDAR2015", metric: "F1 mean (IoU 0.5:0.95)" },
      { name: "TotalText", metric: "F1 mean (IoU 0.5:0.95)" },
      { name: "SROIE", metric: "F1 mean (IoU 0.5:0.95)" }
    ],
    [
      [30.5, 25.6, 25.7, 58.6],
      [11.0, 8.5, 20.5, 18.6],
      [22.7, 30.0, 33.0, 40.4],
      [25.7, 27.1, 35.4, 43.3],
      [8.0, 7.8, 16.6, 7.1],
      [10.0, 13.4, 30.8, 16.7],
      [28.0, 28.1, 40.6, 44.8],
      [29.4, 24.1, 43.3, 40.3],
      [30.5, 42.7, 42.5, 62.1],
      [39.0, 47.8, 52.0, 68.1]
    ]
  );

  const seg = table(
    [
      ["LISA-13B (ft)", "spec"], ["GSVA-7B", "spec"], ["LaSagnA-7B", "spec"], ["Text4Seg-7B", "spec"],
      ["VRS-HQ-7B", "spec"], ["UniPixel-7B", "vlm"], ["Youtu-VL-4B", "vlm"], OURS2, OURS4
    ],
    [
      { name: "RefCOCO val", metric: "cIoU" },
      { name: "RefCOCO testA", metric: "cIoU" },
      { name: "RefCOCO testB", metric: "cIoU" },
      { name: "RefCOCO+ val", metric: "cIoU" },
      { name: "RefCOCO+ testA", metric: "cIoU" },
      { name: "RefCOCO+ testB", metric: "cIoU" },
      { name: "RefCOCOg val", metric: "cIoU" },
      { name: "RefCOCOg test", metric: "cIoU" },
      { name: "ReasonSeg val", metric: "gIoU" },
      { name: "ReasonSeg test", metric: "gIoU" },
      { name: "Ref-ZOM", metric: "cIoU" },
      { name: "Ref-ZOM Rej", metric: "Rejection rate (%)" }
    ],
    [
      [76.0, 78.8, 72.9, 65.0, 70.2, 58.1, 69.5, 70.5, 65.0, 61.3, null, null],
      [77.2, 78.9, 73.5, 65.9, 69.6, 59.8, 72.7, 73.3, 50.5, null, 66.6, 41.9],
      [76.8, 78.7, 73.8, 66.4, 70.6, 60.1, 70.6, 71.9, null, null, 57.4, 0.7],
      [79.3, 81.9, 76.2, 72.1, 77.6, 66.1, 72.1, 73.9, null, null, 54.5, 25.9],
      [73.5, 77.5, 69.5, 61.7, 67.6, 54.3, 66.7, 67.5, 55.2, 51.7, 63.0, 13.9],
      [80.8, 83.0, 77.4, 75.3, 80.1, 70.0, 76.4, 77.1, 60.5, null, 62.9, 0.3],
      [80.7, 82.0, 78.4, 76.2, 79.6, 71.4, 76.5, 76.6, null, null, 61.0, 1.3],
      [79.7, 82.9, 78.1, 73.1, 79.0, 66.3, 74.4, 74.9, 60.9, 56.9, 73.4, 62.4],
      [81.4, 84.5, 80.1, 76.5, 81.1, 72.8, 77.7, 77.8, 71.2, 66.8, 72.2, 86.0]
    ]
  );

  const boxMat = [
    ["Matte Anything", "spec"], ["SAM2-Matting (SAM2.1-T)", "spec"],
    ["SAM2-Matting (SAM2.1-B+)", "spec"], ["SAM2-Matting (SAM3)", "spec"], OURS2, OURS4
  ];
  const textMat = [
    ["SAM2-Matting (SAM3)", "spec"], ["Qwen3-VL-8B + SAM2-Matting", "gen"],
    ["Qwen3.5-9B + SAM2-Matting", "gen"], OURS2, OURS4
  ];
  const mat = []
    .concat(table(boxMat, [{ name: "P3M-500-NP · box", metric: "MAD ×10³ (box prompt)", lower: true }],
      [[24.4], [22.7], [19.0], [15.4], [6.6], [7.3]]))
    .concat(table(boxMat, [{ name: "AM-2K · box", metric: "MAD ×10³ (box prompt)", lower: true }],
      [[13.2], [11.0], [11.1], [12.1], [5.7], [5.0]]))
    .concat(table(boxMat, [{ name: "PPM-100 · box", metric: "MAD ×10³ (box prompt)", lower: true }],
      [[10.5], [9.7], [9.3], [10.8], [5.3], [6.1]]))
    .concat(table(textMat, [{ name: "WeRefMat · category", metric: "MAD ×10³ (category prompt)", lower: true }],
      [[70.8], [80.7], [77.8], [17.7], [16.4]]))
    .concat(table(textMat, [{ name: "WeRefMat · referring", metric: "MAD ×10³ (referring prompt)", lower: true }],
      [[51.9], [23.0], [22.4], [15.8], [9.8]]));

  window.BENCH = [
    { id: "det", label: "Detection", items: det, default: 2 },
    { id: "ref", label: "Referring", items: ref, default: 3 },
    { id: "cnt", label: "Counting", items: cnt, default: 7 },
    { id: "pt", label: "Pointing", items: pt, default: 0 },
    { id: "ocr", label: "OCR", items: ocr, default: 1 },
    { id: "seg", label: "Referring Seg.", items: seg, default: 11 },
    { id: "mat", label: "Matting", items: mat, default: 1 }
  ];
})();
