# Examples

`200021086_00026_2.jpg` — page 26 (image 2) of CODH manuscript 200021086, 伊曾保物語, from the CODH Kuzushiji dataset
(http://codh.rois.ac.jp/char-shape/book/200021086/), CC BY-SA 4.0, © 国文学研究資料館 / CODH. Outputs of the page reader on this page: `200021086_00026_2.expected.json` (classifier `vit-v2-epoch24`) and
`200021086_00026_2.expected-epoch06.json` (`vit-v2-epoch06`); detector `yolo11m-kuzu-v6`, imgsz 1024, conf 0.25, τ = 0, float32 on CPU.

`crops/` holds six CODH character crops with the outputs of both checkpoints in `crops.expected.json` (epoch 24) and
`crops.expected-epoch06.json` (epoch 6). The 60 crops of the numerical tolerance check are in `../fixtures/`.
